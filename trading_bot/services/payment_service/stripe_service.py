"""
Stripe payment service for handling subscriptions and payments
"""
import logging
import os
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StripeService:
    """Service to handle Stripe payments and subscriptions"""
    
    def __init__(self, db=None):
        """
        Initialize the Stripe service
        
        Args:
            db: Database connection
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing StripeService (minimal implementation)")
        
        self.db = db
        
        # Stripe API keys
        self.live_secret_key = os.environ.get("STRIPE_LIVE_SECRET_KEY", "")
        self.test_secret_key = os.environ.get("STRIPE_TEST_SECRET_KEY", "")
        self.webhook_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
        
        self.logger.info(f"Stripe API configuration loaded: {'LIVE' if self.live_secret_key else 'TEST'} mode")
        
        try:
            import stripe
            self.stripe = stripe
            self.stripe.api_key = self.live_secret_key or self.test_secret_key
            self.logger.info("Stripe library imported successfully")
        except ImportError:
            self.logger.warning("Failed to import stripe library. Payment features will be limited.")
            self.stripe = None
    
    async def create_checkout_session(self, user_id: int, price_id: str, success_url: str, cancel_url: str) -> Optional[str]:
        """
        Create a Stripe checkout session
        
        Args:
            user_id: Telegram user ID
            price_id: Stripe price ID
            success_url: URL to redirect on success
            cancel_url: URL to redirect on cancel
            
        Returns:
            Optional[str]: Checkout URL or None if creation failed
        """
        self.logger.info(f"Creating checkout session for user {user_id} with price ID {price_id}")
        
        if not self.stripe:
            self.logger.error("Stripe library not available")
            return None
        
        try:
            # Create checkout session
            session = self.stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price': price_id,
                    'quantity': 1,
                }],
                mode='subscription',
                success_url=success_url,
                cancel_url=cancel_url,
                client_reference_id=str(user_id),
            )
            
            # Store session in database if available
            if self.db:
                await self.db.save_checkout_session(user_id, session.id, price_id)
            
            return session.url
        except Exception as e:
            self.logger.error(f"Error creating checkout session: {str(e)}")
            return None
    
    async def handle_webhook_event(self, payload: Dict[str, Any], signature: str) -> Dict[str, Any]:
        """
        Handle a Stripe webhook event
        
        Args:
            payload: Webhook event payload
            signature: Stripe signature header
            
        Returns:
            Dict[str, Any]: Processing result
        """
        self.logger.info("Processing Stripe webhook event")
        
        if not self.stripe:
            self.logger.error("Stripe library not available")
            return {"status": "error", "message": "Stripe library not available"}
        
        try:
            # Verify the event
            if not self.webhook_secret:
                self.logger.warning("No webhook secret configured, skipping signature verification")
                event = payload
            else:
                event = self.stripe.Webhook.construct_event(
                    payload, signature, self.webhook_secret
                )
            
            # Process the event
            event_type = event.get('type')
            self.logger.info(f"Processing webhook event: {event_type}")
            
            # Handle checkout session completed
            if event_type == 'checkout.session.completed':
                session = event.get('data', {}).get('object', {})
                user_id = session.get('client_reference_id')
                
                if not user_id:
                    self.logger.warning("No client_reference_id in session")
                    return {"status": "error", "message": "No user ID"}
                
                # Convert user_id to int
                try:
                    user_id = int(user_id)
                except (ValueError, TypeError):
                    self.logger.error(f"Invalid user ID: {user_id}")
                    return {"status": "error", "message": "Invalid user ID"}
                
                # Store subscription data
                subscription_id = session.get('subscription')
                
                if subscription_id and self.db:
                    # Get subscription details
                    subscription = self.stripe.Subscription.retrieve(subscription_id)
                    
                    # Calculate start and end dates
                    start_date = datetime.fromtimestamp(subscription.current_period_start)
                    end_date = datetime.fromtimestamp(subscription.current_period_end)
                    
                    # Save subscription to database
                    await self.db.save_user_subscription(
                        user_id=user_id,
                        subscription_type="monthly",
                        start_date=start_date,
                        end_date=end_date,
                        subscription_id=subscription_id
                    )
                    
                    self.logger.info(f"Subscription created for user {user_id}")
                    
                    return {"status": "success", "message": "Subscription created", "user_id": user_id}
                else:
                    self.logger.warning(f"No subscription ID in session or database not available")
                    return {"status": "error", "message": "No subscription ID or database not available"}
            
            # Handle subscription payment failed
            elif event_type == 'invoice.payment_failed':
                invoice = event.get('data', {}).get('object', {})
                subscription_id = invoice.get('subscription')
                
                if subscription_id and self.db:
                    # Flag payment as failed
                    await self.db.set_payment_failed_by_subscription(subscription_id)
                    
                    self.logger.info(f"Payment failed for subscription {subscription_id}")
                    
                    return {"status": "success", "message": "Payment failed status set", "subscription_id": subscription_id}
                else:
                    self.logger.warning(f"No subscription ID in invoice or database not available")
                    return {"status": "error", "message": "No subscription ID or database not available"}
            
            # Handle other events
            return {"status": "success", "message": f"Event {event_type} processed but not handled"}
            
        except Exception as e:
            self.logger.error(f"Error processing webhook event: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    async def get_customer_subscription(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a customer's subscription
        
        Args:
            user_id: Telegram user ID
            
        Returns:
            Optional[Dict[str, Any]]: Subscription details or None
        """
        self.logger.info(f"Getting subscription for user {user_id}")
        
        if not self.db:
            self.logger.error("Database not available")
            return None
        
        try:
            # Get subscription from database
            subscription_data = await self.db.get_user_subscription(user_id)
            
            if not subscription_data:
                self.logger.info(f"No subscription found for user {user_id}")
                return None
            
            subscription_id = subscription_data.get('subscription_id')
            
            if not subscription_id or not self.stripe:
                return subscription_data
            
            # Get subscription from Stripe
            try:
                subscription = self.stripe.Subscription.retrieve(subscription_id)
                
                # Update subscription data with Stripe data
                subscription_data['status'] = subscription.status
                subscription_data['cancel_at_period_end'] = subscription.cancel_at_period_end
                subscription_data['current_period_start'] = datetime.fromtimestamp(subscription.current_period_start)
                subscription_data['current_period_end'] = datetime.fromtimestamp(subscription.current_period_end)
            except Exception as stripe_error:
                self.logger.warning(f"Error retrieving subscription from Stripe: {str(stripe_error)}")
            
            return subscription_data
            
        except Exception as e:
            self.logger.error(f"Error getting subscription: {str(e)}")
            return None

    async def handle_subscription_created(self, event_data: Dict[str, Any]) -> bool:
        """Verwerk een subscription.created event van Stripe"""
        try:
            subscription = event_data['object']
            customer_id = subscription.get('customer')
            subscription_id = subscription.get('id')
            status = subscription.get('status')
            
            # Haal de user_id op uit de metadata van de checkout session
            if 'metadata' in subscription and 'user_id' in subscription['metadata']:
                user_id = int(subscription['metadata']['user_id'])
            else:
                # Als de user_id niet in de metadata staat, moeten we de checkout session opzoeken
                checkout_session_id = subscription.get('metadata', {}).get('checkout_session_id')
                if checkout_session_id:
                    session = self.stripe.checkout.Session.retrieve(checkout_session_id)
                    user_id = int(session.get('client_reference_id', 0))
                else:
                    logger.error(f"Could not find user_id for subscription {subscription_id}")
                    return False
            
            # Bepaal het abonnementstype op basis van het product
            plan_type = 'basic'  # Standaard
            if 'items' in subscription and 'data' in subscription['items']:
                for item in subscription['items']['data']:
                    if 'price' in item:
                        price_id = item['price']['id']
                        # Hier zou je een mapping moeten hebben van price_id naar plan_type
                        if price_id == self.stripe.get_price_id('premium'):
                            plan_type = 'premium'
                        elif price_id == self.stripe.get_price_id('pro'):
                            plan_type = 'pro'
            
            # Bereken de einddatum van de periode
            current_period_end = datetime.datetime.fromtimestamp(
                subscription.get('current_period_end', 0), 
                tz=datetime.timezone.utc
            )
            
            # Update de database
            success = await self.db.create_or_update_subscription(
                user_id=user_id,
                stripe_customer_id=customer_id,
                stripe_subscription_id=subscription_id,
                status=status,
                subscription_type=plan_type,
                current_period_end=current_period_end
            )
            
            if success:
                logger.info(f"Updated subscription for user {user_id}: {status}")
                return True
            else:
                logger.error(f"Failed to update subscription for user {user_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error handling subscription created: {str(e)}")
            return False
    
    async def handle_subscription_updated(self, event_data: Dict[str, Any]) -> bool:
        """Verwerk een subscription.updated event van Stripe"""
        # Vergelijkbare logica als handle_subscription_created
        try:
            subscription = event_data['object']
            subscription_id = subscription.get('id')
            status = subscription.get('status')
            
            # Haal het abonnement op uit de database op basis van stripe_subscription_id
            # In een echte implementatie zou je hier een query moeten doen
            # Voor nu doen we een dummyrequest naar de Stripe API om de klant te vinden
            user_subscription = None
            
            # Haal de user_id op uit de metadata van de abonnement
            if 'metadata' in subscription and 'user_id' in subscription['metadata']:
                user_id = int(subscription['metadata']['user_id'])
            else:
                # Als alternatief, zoek in de database op subscription_id
                # Dit is een dummy-implementatie; in werkelijkheid zou je hier een database-query doen
                logger.warning(f"User ID not found in metadata for subscription {subscription_id}")
                return False
            
            # Bereken de einddatum van de periode
            current_period_end = datetime.datetime.fromtimestamp(
                subscription.get('current_period_end', 0), 
                tz=datetime.timezone.utc
            )
            
            # Update de database
            await self.db.create_or_update_subscription(
                user_id=user_id,
                stripe_subscription_id=subscription_id,
                status=status,
                current_period_end=current_period_end
            )
            
            logger.info(f"Updated subscription {subscription_id} status to {status}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling subscription updated: {str(e)}")
            return False
    
    async def handle_payment_failed(self, event_data: Dict[str, Any]) -> bool:
        """Verwerk een payment_intent.payment_failed of invoice.payment_failed event"""
        try:
            # Voeg meer robuuste error handling toe
            if not event_data or not event_data.get('customer'):
                logger.error("Invalid payment failed event data")
                return False
            
            # Voeg meer logging toe voor productie debugging
            logger.error(f"Payment failed for customer {event_data['customer']}: {event_data.get('failure_message')}")
            
            # Haal de klant-ID op
            customer_id = event_data.get('customer')
            if not customer_id:
                logger.warning("No customer ID found in payment failed event")
                return False
            
            # Zoek de gebruiker op basis van customer_id
            user_subscription = await self.db.get_subscription_by_customer(customer_id)
            if not user_subscription:
                logger.warning(f"No user found for customer {customer_id}")
                return False
            
            user_id = user_subscription.get('user_id')
            
            # Update subscription status to inactive
            await self.db.create_or_update_subscription(
                user_id=user_id,
                status='inactive',
                current_period_end=datetime.now(datetime.timezone.utc)  # Set end date to now
            )
            
            # Notify the user of the payment failure
            if hasattr(self, 'telegram_service') and self.telegram_service:
                failed_message = """
❌ <b>Payment Failed - Subscription Inactive</b>

We were unable to process your payment and your subscription is now inactive.

To continue using SigmaPips Trading Bot, please update your payment method:

<b>👉 <a href="https://buy.stripe.com/9AQcPf3j63HL5JS145">Click Here to Reactivate Your Subscription</a></b>

Need help? Type /help for assistance.
"""
                await self.telegram_service.send_message_to_user(user_id, failed_message, parse_mode=ParseMode.HTML)
            
            logger.info(f"Processed payment failure for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error handling payment failed: {str(e)}")
            return False
    
    async def cancel_subscription(self, user_id: int) -> bool:
        """Annuleer een abonnement voor een gebruiker"""
        try:
            # Haal het abonnement op
            subscription_data = await self.db.get_user_subscription(user_id)
            
            if not subscription_data or not subscription_data.get('stripe_subscription_id'):
                logger.warning(f"No active subscription found for user {user_id}")
                return False
            
            subscription_id = subscription_data['stripe_subscription_id']
            
            # Annuleer het abonnement in Stripe
            self.stripe.Subscription.delete(subscription_id)
            
            # Update de database
            await self.db.create_or_update_subscription(
                user_id=user_id,
                status='canceled'
            )
            
            logger.info(f"Canceled subscription for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error canceling subscription: {str(e)}")
            return False
    
    async def handle_checkout_completed(self, event_data: Dict[str, Any]) -> bool:
        """Process a checkout.session.completed event"""
        try:
            session = event_data.get('object', {})
            client_reference_id = session.get('client_reference_id')
            
            if not client_reference_id:
                if session.get('metadata') and session['metadata'].get('user_id'):
                    client_reference_id = session['metadata']['user_id']
            
            if not client_reference_id:
                logger.error("No user ID found in checkout session data")
                return False
            
            # Convert to int if needed
            user_id = int(client_reference_id)
            
            # Register the customer ID if this is a new customer
            customer_id = session.get('customer')
            subscription_id = session.get('subscription')
            
            if customer_id:
                # Update user's subscription in database
                await self.db.create_or_update_subscription(
                    user_id=user_id,
                    stripe_customer_id=customer_id,
                    stripe_subscription_id=subscription_id,
                    status='trialing',  # Start with trial status
                    current_period_end=datetime.now(datetime.timezone.utc) + timedelta(days=14)  # 14-day trial
                )
                
                # Send welcome and instruction message to user
                if hasattr(self, 'telegram_service') and self.telegram_service:
                    welcome_message = """
✅ <b>Thank You for Subscribing to SigmaPips Trading Bot!</b> ✅

Your 14-day FREE trial has been successfully activated. You now have full access to all features and trading signals.

<b>🚀 HOW TO USE:</b>

<b>1. Start with /menu</b>
   • This will show you the main options:
   • <b>Analyze Market</b> - For all market analysis tools
   • <b>Trading Signals</b> - To manage your trading signals

<b>2. Analyze Market options:</b>
   • <b>Technical Analysis</b> - Charts and price levels
   • <b>Market Sentiment</b> - Indicators and market mood
   • <b>Economic Calendar</b> - Upcoming economic events

<b>3. Trading Signals:</b>
   • Set up which signals you want to receive
   • Signals will be sent automatically
   • Each includes entry, stop loss, and take profit levels

Type /menu to start using the bot.
"""
                    # Stuur alleen het welkomstbericht, geen menu of bevestiging
                    await self.telegram_service.send_message_to_user(user_id, welcome_message, parse_mode=ParseMode.HTML)
                    return True
                
                return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error handling checkout completed: {str(e)}")
            return False

    async def simulate_payment_event(self, event_type="payment_intent.succeeded"):
        """Simuleer een Stripe betaalgebeurtenis"""
        # Simuleer de payload van de gebeurtenis
        event_data = {
            "id": f"evt_test_{int(time.time())}",
            "type": event_type,
            "data": {
                "object": {
                    "id": f"pi_test_{int(time.time())}",
                    "customer": "cus_test123",
                    "amount": 2999,  # €29.99 in centen
                    "status": "succeeded",
                    "metadata": {
                        "user_id": "123456789"  # Voeg hier een echte gebruikers-ID in
                    }
                }
            }
        }
        
        # Verwerk de gesimuleerde gebeurtenis
        db = Database()
        success = await self.process_payment_event(event_data, db)
        
        return success, event_data

    async def create_update_payment_session(self, user_id: int) -> str:
        """Maak een sessie om betalingsgegevens bij te werken"""
        try:
            # Haal gebruikersabonnement op
            subscription = await self.db.get_user_subscription(user_id)
            if not subscription:
                logger.error(f"Geen abonnement gevonden voor gebruiker {user_id}")
                return ""
            
            # Haal stripe customer ID en subscription ID op
            customer_id = subscription.get('stripe_customer_id')
            subscription_id = subscription.get('stripe_subscription_id')
            
            if not customer_id or not subscription_id:
                logger.error(f"Ontbrekende Stripe IDs voor gebruiker {user_id}")
                return ""
            
            # Maak een update payment session
            session = self.stripe.billing_portal.Session.create(
                customer=customer_id,
                return_url=f"https://t.me/SignapipsAI_bot?start=return_from_payment"
            )
            
            return session.url
        except Exception as e:
            logger.error(f"Error creating update payment session: {str(e)}")
            return ""

    # Implementeer de andere handler methodes... 
