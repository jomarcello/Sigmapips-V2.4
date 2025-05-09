"""
Stripe configuration module.

This module contains configuration constants and helper functions for Stripe integration.
"""
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Define plan IDs - replace with your actual price IDs
# These are placeholder values and should be updated for production
PRICE_ID_MONTHLY = os.environ.get("STRIPE_PRICE_ID_MONTHLY", "price_1NdLHEFxtP7Bp5a6UVYkD9l7")
PRICE_ID_YEARLY = os.environ.get("STRIPE_PRICE_ID_YEARLY", "price_1NdLHEFxtP7Bp5a6dN1BxQXb")

def get_price_id(plan_type: str = "monthly") -> str:
    """
    Get the Stripe price ID for the given plan type
    
    Args:
        plan_type: The type of plan (monthly or yearly)
        
    Returns:
        str: The Stripe price ID
    """
    if plan_type == "monthly":
        return PRICE_ID_MONTHLY
    elif plan_type == "yearly":
        return PRICE_ID_YEARLY
    else:
        logger.warning(f"Unknown plan type: {plan_type}, using monthly")
        return PRICE_ID_MONTHLY

def get_subscription_features(plan_type: str = "monthly") -> Dict[str, Any]:
    """
    Get the features for the given subscription plan type
    
    Args:
        plan_type: The type of plan (monthly, yearly)
        
    Returns:
        Dict[str, Any]: The features for the given plan
    """
    # Define features for each plan type
    if plan_type == "monthly":
        return {
            "price": "$29.99/month",
            "trial_days": 14,
            "signals": True,
            "analysis": True,
            "max_instruments": 10
        }
    elif plan_type == "yearly":
        return {
            "price": "$299.99/year",
            "trial_days": 14,
            "signals": True,
            "analysis": True,
            "max_instruments": 20
        }
    else:
        logger.warning(f"Unknown plan type: {plan_type}, using monthly")
        return get_subscription_features("monthly")
