import asyncio
import logging
import traceback
from typing import Dict, List, Any, Optional, Callable
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
from telegram.ext import (
    ContextTypes, CommandHandler, CallbackQueryHandler, MessageHandler,
    filters, ConversationHandler
)

from trading_bot.services.chart_service.chart import ChartService
from trading_bot.services.telegram_service.telegram_formatters import format_technical_analysis, format_market_chart
from trading_bot.utils.constants import SIGNAL_SERVICES, SIGNAL_TIMEFRAMES
from trading_bot.utils.helpers import get_user_subscription_status, is_user_subscribed
from trading_bot.services.telegram_service.constants import (
    SIGNAL_ACTION, SIGNAL_BACK, SIGNAL_CHART, SIGNAL_ANALYSIS
)

# Initialize logger
logger = logging.getLogger(__name__)

# Conversation states for Signal flow
(
    SIGNAL_SERVICE_SELECTION,
    SIGNAL_TIMEFRAME_SELECTION,
    SIGNAL_ANALYSIS_OPTIONS,
) = range(3)

# Callback data prefixes for Signal flow to keep it separated from Menu flow
SIGNAL_PREFIX = "signal_"
SIGNAL_SERVICE = f"{SIGNAL_PREFIX}service_"
SIGNAL_TIMEFRAME = f"{SIGNAL_PREFIX}timeframe_"
SIGNAL_ACTION = f"{SIGNAL_PREFIX}action_"

# Other constants
SIGNAL_BACK = f"{SIGNAL_PREFIX}back"
SIGNAL_ANALYZE = f"{SIGNAL_PREFIX}analyze"
SIGNAL_CHART = f"{SIGNAL_PREFIX}chart"

class SignalFlow:
    """
    Signal flow class for handling signal analysis flow separate from menu flow.
    This flow is specifically for signal analysis and cannot be used to select instruments.
    """
    
    def __init__(self, main_bot):
        self.main_bot = main_bot
        self.service_handlers = {}
        
        # Initialize chart service
        self.chart_service = ChartService()

    def get_signal_flows(self) -> Dict[str, CommandHandler]:
        # We don't need any specific command handlers since the signal flow is embedded
        # in the main bot's callback query handler
        return {}
    
    def get_signal_callback_handlers(self) -> List[CallbackQueryHandler]:
        """Get the signal flow callback query handlers."""
        handlers = []
        
        # Add specific callback data handler
        handlers.append(CallbackQueryHandler(self.handle_signal_menu, pattern=f"^{SIGNAL_ACTION}.*"))
        
        # Add signal analysis handlers
        for service in SIGNAL_SERVICES:
            service_key = service.lower().replace(" ", "_")
            callback_handler = CallbackQueryHandler(
                self.get_service_handler(service),
                pattern=f"^{SIGNAL_ACTION}{service_key}$"
            )
            handlers.append(callback_handler)
        
        return handlers
    
    def get_service_handler(self, service: str) -> Callable:
        """Get or create a handler for a specific service."""
        service_key = service.lower().replace(" ", "_")
        
        if service_key not in self.service_handlers:
            # Create a new handler for this service
            async def service_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
                query = update.callback_query
                await query.answer(f"Getting {service} information...")
                
                # Extract symbol and timeframe from context
                symbol = context.user_data.get("signal_symbol", "")
                signal_data = context.user_data.get("signal_data", {})
                
                if not symbol:
                    await query.edit_message_text(
                        text="⚠️ Error: No signal information available.",
                        reply_markup=InlineKeyboardMarkup([[
                            InlineKeyboardButton("↩️ Back to Menu", callback_data="menu")
                        ]])
                    )
                    return
                
                # Build keyboard for back navigation
                keyboard = [[
                    InlineKeyboardButton("↩️ Back", callback_data=f"{SIGNAL_BACK}")
                ]]
                
                try:
                    if service_key == "analysis":
                        # Show technical analysis for this signal
                        timeframe = signal_data.get("timeframe", "1h")
                        
                        # Check subscription status for this feature
                        subscription_status = await get_user_subscription_status(query.from_user.id)
                        if not is_user_subscribed(subscription_status):
                            # User is not subscribed, show subscription message
                            await query.edit_message_text(
                                text=f"⭐️ <b>Premium Feature</b> ⭐️\n\nTechnical Analysis is a premium feature. Please subscribe to access this feature.",
                                parse_mode=ParseMode.HTML,
                                reply_markup=InlineKeyboardMarkup([
                                    [InlineKeyboardButton("💳 Subscribe Now", callback_data="subscription")],
                                    [InlineKeyboardButton("↩️ Back", callback_data=f"{SIGNAL_BACK}")]
                                ])
                            )
                            return
                            
                        # Update message to indicate loading
                        await query.edit_message_text(f"Loading technical analysis for {symbol} ({timeframe})...")
                        
                        # Get technical analysis
                        try:
                            # Haal technische analyse op
                            analysis_text = await self.chart_service.get_technical_analysis(symbol, timeframe)
                            
                            # Send analysis text
                            await context.bot.edit_message_text(
                                chat_id=query.message.chat_id,
                                message_id=query.message.message_id,
                                text=analysis_text,
                                parse_mode=ParseMode.HTML,
                                reply_markup=InlineKeyboardMarkup([
                                    [InlineKeyboardButton("📊 View Chart", callback_data=f"{SIGNAL_CHART}")],
                                    [InlineKeyboardButton("↩️ Back", callback_data=f"{SIGNAL_BACK}")]
                                ])
                            )
                            
                        except Exception as e:
                            logger.error(f"Error getting technical analysis: {str(e)}")
                            logger.error(traceback.format_exc())
                            
                            # Show error message
                            await query.edit_message_text(
                                text=f"⚠️ Error getting technical analysis for {symbol}: {str(e)}",
                                reply_markup=InlineKeyboardMarkup([
                                    [InlineKeyboardButton("🔄 Try Again", callback_data=f"{SIGNAL_ACTION}analysis")],
                                    [InlineKeyboardButton("↩️ Back", callback_data=f"{SIGNAL_BACK}")]
                                ])
                            )
                    elif service_key == "chart":
                        # Show chart for this signal
                        timeframe = signal_data.get("timeframe", "1h")
                        
                        # Check subscription status for this feature
                        subscription_status = await get_user_subscription_status(query.from_user.id)
                        if not is_user_subscribed(subscription_status):
                            # User is not subscribed, show subscription message
                            await query.edit_message_text(
                                text=f"⭐️ <b>Premium Feature</b> ⭐️\n\nChart viewing is a premium feature. Please subscribe to access this feature.",
                                parse_mode=ParseMode.HTML,
                                reply_markup=InlineKeyboardMarkup([
                                    [InlineKeyboardButton("💳 Subscribe Now", callback_data="subscription")],
                                    [InlineKeyboardButton("↩️ Back", callback_data=f"{SIGNAL_BACK}")]
                                ])
                            )
                            return
                            
                        # Call the main bot's show_chart method
                        await self.main_bot.show_chart(
                            update=update,
                            context=context,
                            instrument=symbol,
                            timeframe=timeframe,
                            callback_data=f"{SIGNAL_BACK}"
                        )
                        return
                    else:
                        # Handle other services
                        await query.edit_message_text(
                            text=f"Service '{service}' is not yet implemented.",
                            reply_markup=InlineKeyboardMarkup(keyboard)
                        )
                
                except Exception as e:
                    logger.error(f"Error in service handler: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    # Show error message
                    await query.edit_message_text(
                        text=f"⚠️ Error: {str(e)}",
                        reply_markup=InlineKeyboardMarkup(keyboard)
                    )
            
            self.service_handlers[service_key] = service_handler
        
        return self.service_handlers[service_key]
    
    async def handle_signal_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle the signal menu."""
        query = update.callback_query
        await query.answer()
        
        # Process callback data
        callback_data = query.data
        
        if callback_data == f"{SIGNAL_ACTION}menu":
            # Show signal menu with analysis, chart, etc.
            # Extract symbol and timeframe from context
            symbol = context.user_data.get("signal_symbol", "")
            signal_data = context.user_data.get("signal_data", {})
            
            if not symbol:
                await query.edit_message_text(
                    text="⚠️ Error: No signal information available.",
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("↩️ Back to Menu", callback_data="menu")
                    ]])
                )
                return
            
            # Build menu keyboard
            keyboard = [
                [InlineKeyboardButton("📊 Technical Analysis", callback_data=f"{SIGNAL_ACTION}analysis"),
                 InlineKeyboardButton("📈 Chart", callback_data=f"{SIGNAL_ACTION}chart")],
                [InlineKeyboardButton("↩️ Back to Signal", callback_data=f"{SIGNAL_BACK}")]
            ]
            
            # Send menu
            await query.edit_message_text(
                text=f"🔍 Analysis options for signal {symbol}:",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        
        elif callback_data == f"{SIGNAL_BACK}":
            # Go back to the main signal display
            # Call the main bot's redisplay_signal method
            await self.main_bot.redisplay_signal(update, context)
        
        else:
            # Forward to specific service handlers
            for service in SIGNAL_SERVICES:
                service_key = service.lower().replace(" ", "_")
                if callback_data == f"{SIGNAL_ACTION}{service_key}":
                    handler = self.get_service_handler(service)
                    await handler(update, context)
                    return
            
            # If no handler matched, show the signal menu
            await self.handle_signal_menu(update, context)

    @staticmethod
    async def signal_analysis_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """
        Start the signal analysis flow.
        This is an entry point specifically for signals.
        """
        try:
            # Store the original signal information
            if not context.user_data.get("signal_data"):
                # If no signal data (should not happen), just return
                await update.callback_query.answer("No signal data available")
                return ConversationHandler.END
            
            # Get the symbol from signal data
            symbol = context.user_data["signal_data"].get("symbol", "")
            
            if not symbol:
                await update.callback_query.answer("No symbol found in signal")
                return ConversationHandler.END
            
            # Store the symbol
            context.user_data["signal_symbol"] = symbol
            
            # Create services keyboard
            keyboard = []
            
            # Create row for each service pair
            services_list = SIGNAL_SERVICES.copy()
            for i in range(0, len(services_list), 2):
                row = []
                # Add first service
                service_name = services_list[i]
                row.append(InlineKeyboardButton(
                    service_name, 
                    callback_data=f"{SIGNAL_SERVICE}{service_name.lower()}"
                ))
                
                # Add second service if available
                if i + 1 < len(services_list):
                    service_name = services_list[i + 1]
                    row.append(InlineKeyboardButton(
                        service_name, 
                        callback_data=f"{SIGNAL_SERVICE}{service_name.lower()}"
                    ))
                
                keyboard.append(row)
            
            # Add back button
            keyboard.append([
                InlineKeyboardButton("⬅️ Back", callback_data=SIGNAL_BACK)
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # If update has callback_query (from button press)
            if update.callback_query:
                await update.callback_query.answer()
                await update.callback_query.edit_message_text(
                    text=f"Signal Analysis for {symbol}\n\nChoose Service:",
                    reply_markup=reply_markup
                )
            else:
                # Direct message entry
                await update.message.reply_text(
                    text=f"Signal Analysis for {symbol}\n\nChoose Service:",
                    reply_markup=reply_markup
                )
                
            return SIGNAL_SERVICE_SELECTION
            
        except Exception as e:
            logger.error(f"Error in signal_analysis_start: {str(e)}")
            logger.error(traceback.format_exc())
            
            if update.callback_query:
                await update.callback_query.answer("An error occurred")
                await update.callback_query.edit_message_text("Sorry, an error occurred. Please try again.")
            else:
                await update.message.reply_text("Sorry, an error occurred. Please try again.")
            
            return ConversationHandler.END
    
    @staticmethod
    async def signal_service_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle service selection for signal analysis"""
        query = update.callback_query
        await query.answer()
        
        # Get callback data
        callback_data = query.data
        
        # If back button pressed, go back to signal
        if callback_data == SIGNAL_BACK:
            # This should go back to the signal display
            # We use a special function for that
            return await SignalFlow.back_to_signal(update, context)
        
        # Extract service from callback data
        if callback_data.startswith(SIGNAL_SERVICE):
            service = callback_data[len(SIGNAL_SERVICE):]
            context.user_data["signal_service"] = service
            
            # Create timeframes keyboard
            keyboard = []
            
            # Create row for each timeframe pair
            timeframes_list = SIGNAL_TIMEFRAMES.copy()
            for i in range(0, len(timeframes_list), 3):  # 3 timeframes per row
                row = []
                # Process up to 3 timeframes for this row
                for j in range(3):
                    if i + j < len(timeframes_list):
                        timeframe = timeframes_list[i + j]
                        row.append(InlineKeyboardButton(
                            timeframe, 
                            callback_data=f"{SIGNAL_TIMEFRAME}{timeframe.lower()}"
                        ))
                
                keyboard.append(row)
            
            # Add back button
            keyboard.append([
                InlineKeyboardButton("⬅️ Back", callback_data=SIGNAL_BACK)
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Get the symbol
            symbol = context.user_data.get("signal_symbol", "Unknown")
            
            await query.edit_message_text(
                text=f"Signal Analysis for {symbol}\nService: {service}\n\nChoose Timeframe:",
                reply_markup=reply_markup
            )
            
            return SIGNAL_TIMEFRAME_SELECTION
            
        else:
            await query.edit_message_text("Invalid selection. Please try again.")
            return ConversationHandler.END
    
    @staticmethod
    async def signal_timeframe_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle timeframe selection for signal analysis"""
        query = update.callback_query
        await query.answer()
        
        # Get callback data
        callback_data = query.data
        
        # If back button pressed, go back to service selection
        if callback_data == SIGNAL_BACK:
            return await SignalFlow.signal_analysis_start(update, context)
        
        # Extract timeframe from callback data
        if callback_data.startswith(SIGNAL_TIMEFRAME):
            timeframe = callback_data[len(SIGNAL_TIMEFRAME):]
            context.user_data["signal_timeframe"] = timeframe
            
            # Get symbol and service
            symbol = context.user_data.get("signal_symbol", "Unknown")
            service = context.user_data.get("signal_service", "Unknown")
            
            # Create analysis options keyboard
            keyboard = [
                [
                    InlineKeyboardButton("📊 Technical Analysis", callback_data=f"{SIGNAL_ACTION}analysis"),
                    InlineKeyboardButton("📈 Chart", callback_data=f"{SIGNAL_ACTION}chart")
                ],
                [
                    InlineKeyboardButton("⬅️ Back", callback_data=SIGNAL_BACK)
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(
                text=f"Signal Analysis for {symbol}\nService: {service}\nTimeframe: {timeframe}\n\nChoose Option:",
                reply_markup=reply_markup
            )
            
            return SIGNAL_ANALYSIS_OPTIONS
            
        else:
            await query.edit_message_text("Invalid selection. Please try again.")
            return ConversationHandler.END
    
    @staticmethod
    async def signal_analysis_options(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle analysis options for signal analysis"""
        query = update.callback_query
        await query.answer()
        
        # Get callback data
        callback_data = query.data
        
        # If back button pressed, go back to timeframe selection
        if callback_data == SIGNAL_BACK:
            # Get the service selection
            service = context.user_data.get("signal_service", "")
            
            # Create timeframes keyboard
            keyboard = []
            
            # Create row for each timeframe pair
            timeframes_list = SIGNAL_TIMEFRAMES.copy()
            for i in range(0, len(timeframes_list), 3):  # 3 timeframes per row
                row = []
                # Process up to 3 timeframes for this row
                for j in range(3):
                    if i + j < len(timeframes_list):
                        timeframe = timeframes_list[i + j]
                        row.append(InlineKeyboardButton(
                            timeframe, 
                            callback_data=f"{SIGNAL_TIMEFRAME}{timeframe.lower()}"
                        ))
                
                keyboard.append(row)
            
            # Add back button
            keyboard.append([
                InlineKeyboardButton("⬅️ Back", callback_data=SIGNAL_BACK)
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Get the symbol
            symbol = context.user_data.get("signal_symbol", "Unknown")
            
            await query.edit_message_text(
                text=f"Signal Analysis for {symbol}\nService: {service}\n\nChoose Timeframe:",
                reply_markup=reply_markup
            )
            
            return SIGNAL_TIMEFRAME_SELECTION
        
        # Handle the analysis options
        if callback_data.startswith(SIGNAL_ACTION):
            action = callback_data[len(SIGNAL_ACTION):]
            
            # Get stored data
            symbol = context.user_data.get("signal_symbol", "")
            timeframe = context.user_data.get("signal_timeframe", "1h")
            
            if not symbol:
                await query.edit_message_text("Error: No symbol specified. Please try again.")
                return ConversationHandler.END
            
            # Process requested action
            if action == "analysis":
                # Show loading message
                await query.edit_message_text(f"Loading technical analysis for {symbol} ({timeframe})...")
                
                try:
                    # Get technical analysis
                    chart_service = ChartService()
                    analysis_text = await chart_service.get_technical_analysis(symbol, timeframe)
                    
                    # Check for errors
                    if "error" in analysis_text:
                        error_msg = analysis_text.get("message", "Unknown error")
                        keyboard = [[
                            InlineKeyboardButton("🔄 Try Again", callback_data=f"{SIGNAL_ACTION}analysis"),
                            InlineKeyboardButton("⬅️ Back", callback_data=SIGNAL_BACK)
                        ]]
                        reply_markup = InlineKeyboardMarkup(keyboard)
                        
                        await query.edit_message_text(
                            text=f"Error getting analysis for {symbol}: {error_msg}",
                            reply_markup=reply_markup
                        )
                        return SIGNAL_ANALYSIS_OPTIONS
                    
                    # Format analysis
                    formatted_analysis = format_technical_analysis(analysis_text)
                    
                    # Create keyboard for options
                    keyboard = [
                        [
                            InlineKeyboardButton("📈 Show Chart", callback_data=f"{SIGNAL_ACTION}chart"),
                            InlineKeyboardButton("🔄 Refresh", callback_data=f"{SIGNAL_ACTION}analysis")
                        ],
                        [
                            InlineKeyboardButton("⬅️ Back", callback_data=SIGNAL_BACK)
                        ]
                    ]
                    
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    await query.edit_message_text(
                        text=formatted_analysis,
                        reply_markup=reply_markup,
                        parse_mode="Markdown"
                    )
                    
                except Exception as e:
                    logger.error(f"Error getting technical analysis: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    keyboard = [[
                        InlineKeyboardButton("🔄 Try Again", callback_data=f"{SIGNAL_ACTION}analysis"),
                        InlineKeyboardButton("⬅️ Back", callback_data=SIGNAL_BACK)
                    ]]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    await query.edit_message_text(
                        text=f"Error getting analysis for {symbol}: {str(e)}",
                        reply_markup=reply_markup
                    )
                
                return SIGNAL_ANALYSIS_OPTIONS
                
            elif action == "chart":
                # Show loading message
                await query.edit_message_text(f"Generating chart for {symbol} ({timeframe})...")
                
                try:
                    # Get chart data
                    # Use dedicated formatter for chart message
                    chart_message = format_market_chart(symbol, timeframe)
                    
                    # Create keyboard for options
                    keyboard = [
                        [
                            InlineKeyboardButton("📊 Technical Analysis", callback_data=f"{SIGNAL_ACTION}analysis"),
                            InlineKeyboardButton("🔄 Refresh Chart", callback_data=f"{SIGNAL_ACTION}chart")
                        ],
                        [
                            InlineKeyboardButton("⬅️ Back", callback_data=SIGNAL_BACK)
                        ]
                    ]
                    
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    await query.edit_message_text(
                        text=chart_message,
                        reply_markup=reply_markup,
                        parse_mode="Markdown"
                    )
                    
                except Exception as e:
                    logger.error(f"Error generating chart: {str(e)}")
                    logger.error(traceback.format_exc())
                    
                    keyboard = [[
                        InlineKeyboardButton("🔄 Try Again", callback_data=f"{SIGNAL_ACTION}chart"),
                        InlineKeyboardButton("⬅️ Back", callback_data=SIGNAL_BACK)
                    ]]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    await query.edit_message_text(
                        text=f"Error generating chart for {symbol}: {str(e)}",
                        reply_markup=reply_markup
                    )
                
                return SIGNAL_ANALYSIS_OPTIONS
            
        # Invalid action
        await query.edit_message_text("Invalid selection. Please try again.")
        return ConversationHandler.END
    
    @staticmethod
    async def back_to_signal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Return to the original signal display"""
        query = update.callback_query
        
        # Get original signal data
        signal_data = context.user_data.get("signal_data", {})
        
        if not signal_data:
            await query.edit_message_text("Signal data not available. Please try again.")
            return ConversationHandler.END
        
        # Recreate the original signal message and keyboard
        # This will depend on your signal format implementation
        symbol = signal_data.get("symbol", "Unknown")
        message = signal_data.get("message", f"Signal for {symbol}")
        
        # Create a keyboard with signal options
        keyboard = [
            [
                InlineKeyboardButton("📊 Analyze Market", callback_data=f"{SIGNAL_PREFIX}analyze")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            text=message,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )
        
        return ConversationHandler.END
    
    @staticmethod
    def get_handlers() -> List[Any]:
        """
        Get all handlers for the signal flow
        
        Returns:
            List[Any]: List of handlers
        """
        # Create the conversation handler for signal flow
        signal_conv_handler = ConversationHandler(
            entry_points=[
                CallbackQueryHandler(
                    SignalFlow.signal_analysis_start,
                    pattern=f"^{SIGNAL_PREFIX}analyze$"
                )
            ],
            states={
                SIGNAL_SERVICE_SELECTION: [
                    CallbackQueryHandler(
                        SignalFlow.signal_service_selection,
                        pattern=f"^{SIGNAL_SERVICE}|{SIGNAL_BACK}$"
                    )
                ],
                SIGNAL_TIMEFRAME_SELECTION: [
                    CallbackQueryHandler(
                        SignalFlow.signal_timeframe_selection,
                        pattern=f"^{SIGNAL_TIMEFRAME}|{SIGNAL_BACK}$"
                    )
                ],
                SIGNAL_ANALYSIS_OPTIONS: [
                    CallbackQueryHandler(
                        SignalFlow.signal_analysis_options,
                        pattern=f"^{SIGNAL_ACTION}|{SIGNAL_BACK}$"
                    )
                ]
            },
            fallbacks=[
                # Fallback to end conversation
                CommandHandler("cancel", lambda u, c: ConversationHandler.END)
            ],
            name="signal_flow",
            persistent=False
        )
        
        return [signal_conv_handler] 