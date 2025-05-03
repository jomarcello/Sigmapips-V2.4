import asyncio
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ParseMode
from telegram.ext import (
    ContextTypes, CommandHandler, CallbackQueryHandler, MessageHandler,
    filters, ConversationHandler
)

from trading_bot.services.chart_service.chart import ChartService
from trading_bot.services.telegram_service.telegram_formatters import format_technical_analysis, format_market_chart
from trading_bot.utils.constants import INSTRUMENT_CATEGORIES, MARKET_INSTRUMENTS, TIMEFRAMES
from trading_bot.utils.helpers import get_user_subscription_status, is_user_subscribed
from trading_bot.services.telegram_service.constants import (
    MENU_SELECT_MARKET, MENU_SELECT_INSTRUMENT, MENU_SELECT_TIMEFRAME,
    MENU_ANALYSIS, MENU_CHART, MENU_BACK, MENU_CLOSE
)

# Initialize logger
logger = logging.getLogger(__name__)

# Conversation states for Menu flow
(
    MENU_CATEGORY_SELECTION,
    MENU_INSTRUMENT_SELECTION,
    MENU_MARKET_SELECTION,
    MENU_TIMEFRAME_SELECTION,
    MENU_SERVICE_SELECTION,
    MENU_ACTION_SELECTION,
) = range(6)

# Callback data prefixes for Menu flow to keep it separated from Signal flow
MENU_PREFIX = "menu_"
MENU_CATEGORY = f"{MENU_PREFIX}category_"
MENU_INSTRUMENT = f"{MENU_PREFIX}instrument_"
MENU_MARKET = f"{MENU_PREFIX}market_"
MENU_TIMEFRAME = f"{MENU_PREFIX}timeframe_"
MENU_SERVICE = f"{MENU_PREFIX}service_"
MENU_ACTION = f"{MENU_PREFIX}action_"

# Other constants
MENU_BACK = f"{MENU_PREFIX}back"

class MenuFlow:
    """
    Menu flow class for handling menu navigation.
    This flow is for the main menu and instrument selection.
    """
    
    def __init__(self, main_bot):
        self.main_bot = main_bot
        
        # Initialize chart service
        self.chart_service = ChartService()
        
    @staticmethod
    async def menu_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Start the menu flow with categories"""
        try:
            # Clear any existing menu data to start fresh
            context.user_data.pop("menu_category", None)
            context.user_data.pop("menu_instrument", None)
            context.user_data.pop("menu_market", None)
            context.user_data.pop("menu_timeframe", None)
            context.user_data.pop("menu_service", None)
            
            # Create keyboard with categories
            keyboard = []
            
            # Add categories in pairs
            categories = list(INSTRUMENT_CATEGORIES.keys())
            for i in range(0, len(categories), 2):
                row = []
                # Add first category
                category = categories[i]
                row.append(InlineKeyboardButton(
                    category, 
                    callback_data=f"{MENU_CATEGORY}{category.lower()}"
                ))
                
                # Add second category if available
                if i + 1 < len(categories):
                    category = categories[i + 1]
                    row.append(InlineKeyboardButton(
                        category, 
                        callback_data=f"{MENU_CATEGORY}{category.lower()}"
                    ))
                
                keyboard.append(row)
                
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # If update has callback_query (button press)
            if update.callback_query:
                await update.callback_query.answer()
                await update.callback_query.edit_message_text(
                    text="Welcome to the Trading Bot Menu!\n\nSelect a category:",
                    reply_markup=reply_markup
                )
            else:
                # Direct message (/menu command)
                await update.message.reply_text(
                    text="Welcome to the Trading Bot Menu!\n\nSelect a category:",
                    reply_markup=reply_markup
                )
                
            return MENU_CATEGORY_SELECTION
            
        except Exception as e:
            logger.error(f"Error in menu_start: {str(e)}")
            logger.error(traceback.format_exc())
            
            if update.callback_query:
                await update.callback_query.answer("An error occurred")
                await update.callback_query.edit_message_text("Sorry, an error occurred. Please try again.")
            else:
                await update.message.reply_text("Sorry, an error occurred. Please try again.")
            
            return ConversationHandler.END
    
    @staticmethod
    async def menu_category_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle category selection in the menu"""
        query = update.callback_query
        await query.answer()
        
        # Get callback data
        callback_data = query.data
        
        # Extract category from callback data
        if callback_data.startswith(MENU_CATEGORY):
            category = callback_data[len(MENU_CATEGORY):]
            context.user_data["menu_category"] = category
            
            # Get instruments for this category
            if category.lower() in [cat.lower() for cat in INSTRUMENT_CATEGORIES]:
                # Find the exact category key
                category_key = next(cat for cat in INSTRUMENT_CATEGORIES if cat.lower() == category.lower())
                instruments = INSTRUMENT_CATEGORIES[category_key]
                
                # Create keyboard with instruments
                keyboard = []
                
                # Add instruments in pairs
                for i in range(0, len(instruments), 2):
                    row = []
                    # Add first instrument
                    instrument = instruments[i]
                    row.append(InlineKeyboardButton(
                        instrument, 
                        callback_data=f"{MENU_INSTRUMENT}{instrument.lower()}"
                    ))
                    
                    # Add second instrument if available
                    if i + 1 < len(instruments):
                        instrument = instruments[i + 1]
                        row.append(InlineKeyboardButton(
                            instrument, 
                            callback_data=f"{MENU_INSTRUMENT}{instrument.lower()}"
                        ))
                    
                    keyboard.append(row)
                
                # Add back button
                keyboard.append([
                    InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK)
                ])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(
                    text=f"Category: {category_key}\n\nSelect an instrument:",
                    reply_markup=reply_markup
                )
                
                return MENU_INSTRUMENT_SELECTION
            else:
                await query.edit_message_text("Invalid category. Please try again.")
                return ConversationHandler.END
        else:
            await query.edit_message_text("Invalid selection. Please try again.")
            return ConversationHandler.END
    
    @staticmethod
    async def menu_instrument_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle instrument selection in the menu"""
        query = update.callback_query
        await query.answer()
        
        # Get callback data
        callback_data = query.data
        
        # If back button pressed, go back to category selection
        if callback_data == MENU_BACK:
            return await MenuFlow.menu_start(update, context)
        
        # Extract instrument from callback data
        if callback_data.startswith(MENU_INSTRUMENT):
            instrument = callback_data[len(MENU_INSTRUMENT):]
            context.user_data["menu_instrument"] = instrument
            
            # Get markets for this instrument
            markets = []
            
            # Find the markets for the selected instrument
            for market_key, instruments in MARKET_INSTRUMENTS.items():
                if instrument.lower() in [inst.lower() for inst in instruments]:
                    markets.append(market_key)
            
            if markets:
                # Create keyboard with markets
                keyboard = []
                
                # Add markets in pairs
                for i in range(0, len(markets), 2):
                    row = []
                    # Add first market
                    market = markets[i]
                    row.append(InlineKeyboardButton(
                        market, 
                        callback_data=f"{MENU_MARKET}{market.lower()}"
                    ))
                    
                    # Add second market if available
                    if i + 1 < len(markets):
                        market = markets[i + 1]
                        row.append(InlineKeyboardButton(
                            market, 
                            callback_data=f"{MENU_MARKET}{market.lower()}"
                        ))
                    
                    keyboard.append(row)
                
                # Add back button
                keyboard.append([
                    InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK)
                ])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                # Get the category
                category = context.user_data.get("menu_category", "")
                
                await query.edit_message_text(
                    text=f"Category: {category}\nInstrument: {instrument}\n\nSelect a market:",
                    reply_markup=reply_markup
                )
                
                return MENU_MARKET_SELECTION
            else:
                # No markets found for this instrument
                await query.edit_message_text(
                    text=f"No markets found for instrument: {instrument}. Please try again."
                )
                return ConversationHandler.END
        else:
            await query.edit_message_text("Invalid selection. Please try again.")
            return ConversationHandler.END
    
    @staticmethod
    async def menu_market_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle market selection in the menu"""
        query = update.callback_query
        await query.answer()
        
        # Get callback data
        callback_data = query.data
        
        # If back button pressed, go back to instrument selection
        if callback_data == MENU_BACK:
            # Get the category
            category = context.user_data.get("menu_category", "")
            
            if category:
                # Find the exact category key
                category_key = next(cat for cat in INSTRUMENT_CATEGORIES if cat.lower() == category.lower())
                instruments = INSTRUMENT_CATEGORIES[category_key]
                
                # Create keyboard with instruments
                keyboard = []
                
                # Add instruments in pairs
                for i in range(0, len(instruments), 2):
                    row = []
                    # Add first instrument
                    instrument = instruments[i]
                    row.append(InlineKeyboardButton(
                        instrument, 
                        callback_data=f"{MENU_INSTRUMENT}{instrument.lower()}"
                    ))
                    
                    # Add second instrument if available
                    if i + 1 < len(instruments):
                        instrument = instruments[i + 1]
                        row.append(InlineKeyboardButton(
                            instrument, 
                            callback_data=f"{MENU_INSTRUMENT}{instrument.lower()}"
                        ))
                    
                    keyboard.append(row)
                
                # Add back button
                keyboard.append([
                    InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK)
                ])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(
                    text=f"Category: {category_key}\n\nSelect an instrument:",
                    reply_markup=reply_markup
                )
                
                return MENU_INSTRUMENT_SELECTION
            else:
                # If no category, go back to start
                return await MenuFlow.menu_start(update, context)
        
        # Extract market from callback data
        if callback_data.startswith(MENU_MARKET):
            market = callback_data[len(MENU_MARKET):]
            context.user_data["menu_market"] = market
            
            # Create timeframes keyboard
            keyboard = []
            
            # Create row for each timeframe pair
            timeframes_list = TIMEFRAMES.copy()
            for i in range(0, len(timeframes_list), 3):  # 3 timeframes per row
                row = []
                # Process up to 3 timeframes for this row
                for j in range(3):
                    if i + j < len(timeframes_list):
                        timeframe = timeframes_list[i + j]
                        row.append(InlineKeyboardButton(
                            timeframe, 
                            callback_data=f"{MENU_TIMEFRAME}{timeframe.lower()}"
                        ))
                
                keyboard.append(row)
            
            # Add back button
            keyboard.append([
                InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK)
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Get the category and instrument
            category = context.user_data.get("menu_category", "")
            instrument = context.user_data.get("menu_instrument", "")
            
            await query.edit_message_text(
                text=f"Category: {category}\nInstrument: {instrument}\nMarket: {market}\n\nSelect a timeframe:",
                reply_markup=reply_markup
            )
            
            return MENU_TIMEFRAME_SELECTION
            
        else:
            await query.edit_message_text("Invalid selection. Please try again.")
            return ConversationHandler.END
    
    @staticmethod
    async def menu_timeframe_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle timeframe selection in the menu"""
        query = update.callback_query
        await query.answer()
        
        # Get callback data
        callback_data = query.data
        
        # If back button pressed, go back to market selection
        if callback_data == MENU_BACK:
            # Get the category and instrument
            category = context.user_data.get("menu_category", "")
            instrument = context.user_data.get("menu_instrument", "")
            
            # Get markets for this instrument
            markets = []
            
            # Find the markets for the selected instrument
            for market_key, instruments in MARKET_INSTRUMENTS.items():
                if instrument.lower() in [inst.lower() for inst in instruments]:
                    markets.append(market_key)
            
            if markets:
                # Create keyboard with markets
                keyboard = []
                
                # Add markets in pairs
                for i in range(0, len(markets), 2):
                    row = []
                    # Add first market
                    market = markets[i]
                    row.append(InlineKeyboardButton(
                        market, 
                        callback_data=f"{MENU_MARKET}{market.lower()}"
                    ))
                    
                    # Add second market if available
                    if i + 1 < len(markets):
                        market = markets[i + 1]
                        row.append(InlineKeyboardButton(
                            market, 
                            callback_data=f"{MENU_MARKET}{market.lower()}"
                        ))
                    
                    keyboard.append(row)
                
                # Add back button
                keyboard.append([
                    InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK)
                ])
                
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await query.edit_message_text(
                    text=f"Category: {category}\nInstrument: {instrument}\n\nSelect a market:",
                    reply_markup=reply_markup
                )
                
                return MENU_MARKET_SELECTION
            else:
                # If no markets, go back to instrument selection
                return await MenuFlow.menu_instrument_selection(update, context)
        
        # Extract timeframe from callback data
        if callback_data.startswith(MENU_TIMEFRAME):
            timeframe = callback_data[len(MENU_TIMEFRAME):]
            context.user_data["menu_timeframe"] = timeframe
            
            # Create services keyboard
            keyboard = [
                [
                    InlineKeyboardButton("TradingView", callback_data=f"{MENU_SERVICE}tradingview"),
                    InlineKeyboardButton("Yahoo Finance", callback_data=f"{MENU_SERVICE}yahoo")
                ],
                [
                    InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK)
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Get the category, instrument, and market
            category = context.user_data.get("menu_category", "")
            instrument = context.user_data.get("menu_instrument", "")
            market = context.user_data.get("menu_market", "")
            
            await query.edit_message_text(
                text=f"Category: {category}\nInstrument: {instrument}\nMarket: {market}\nTimeframe: {timeframe}\n\nSelect a service:",
                reply_markup=reply_markup
            )
            
            return MENU_SERVICE_SELECTION
            
        else:
            await query.edit_message_text("Invalid selection. Please try again.")
            return ConversationHandler.END
    
    @staticmethod
    async def menu_service_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle service selection in the menu"""
        query = update.callback_query
        await query.answer()
        
        # Get callback data
        callback_data = query.data
        
        # If back button pressed, go back to timeframe selection
        if callback_data == MENU_BACK:
            # Create timeframes keyboard
            keyboard = []
            
            # Create row for each timeframe pair
            timeframes_list = TIMEFRAMES.copy()
            for i in range(0, len(timeframes_list), 3):  # 3 timeframes per row
                row = []
                # Process up to 3 timeframes for this row
                for j in range(3):
                    if i + j < len(timeframes_list):
                        timeframe = timeframes_list[i + j]
                        row.append(InlineKeyboardButton(
                            timeframe, 
                            callback_data=f"{MENU_TIMEFRAME}{timeframe.lower()}"
                        ))
                
                keyboard.append(row)
            
            # Add back button
            keyboard.append([
                InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK)
            ])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Get stored data
            category = context.user_data.get("menu_category", "")
            instrument = context.user_data.get("menu_instrument", "")
            market = context.user_data.get("menu_market", "")
            
            await query.edit_message_text(
                text=f"Category: {category}\nInstrument: {instrument}\nMarket: {market}\n\nSelect a timeframe:",
                reply_markup=reply_markup
            )
            
            return MENU_TIMEFRAME_SELECTION
        
        # Extract service from callback data
        if callback_data.startswith(MENU_SERVICE):
            service = callback_data[len(MENU_SERVICE):]
            context.user_data["menu_service"] = service
            
            # Create action keyboard
            keyboard = [
                [
                    InlineKeyboardButton("📊 Technical Analysis", callback_data=f"{MENU_ACTION}analysis"),
                    InlineKeyboardButton("📈 Chart", callback_data=f"{MENU_ACTION}chart")
                ],
                [
                    InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK)
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Get stored data
            category = context.user_data.get("menu_category", "")
            instrument = context.user_data.get("menu_instrument", "")
            market = context.user_data.get("menu_market", "")
            timeframe = context.user_data.get("menu_timeframe", "")
            
            await query.edit_message_text(
                text=f"Category: {category}\nInstrument: {instrument}\nMarket: {market}\nTimeframe: {timeframe}\nService: {service}\n\nSelect an action:",
                reply_markup=reply_markup
            )
            
            return MENU_ACTION_SELECTION
            
        else:
            await query.edit_message_text("Invalid selection. Please try again.")
            return ConversationHandler.END
    
    @staticmethod
    async def menu_action_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Handle action selection in the menu"""
        query = update.callback_query
        await query.answer()
        
        # Get callback data
        callback_data = query.data
        
        # If back button pressed, go back to service selection
        if callback_data == MENU_BACK:
            # Create services keyboard
            keyboard = [
                [
                    InlineKeyboardButton("TradingView", callback_data=f"{MENU_SERVICE}tradingview"),
                    InlineKeyboardButton("Yahoo Finance", callback_data=f"{MENU_SERVICE}yahoo")
                ],
                [
                    InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK)
                ]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Get stored data
            category = context.user_data.get("menu_category", "")
            instrument = context.user_data.get("menu_instrument", "")
            market = context.user_data.get("menu_market", "")
            timeframe = context.user_data.get("menu_timeframe", "")
            
            await query.edit_message_text(
                text=f"Category: {category}\nInstrument: {instrument}\nMarket: {market}\nTimeframe: {timeframe}\n\nSelect a service:",
                reply_markup=reply_markup
            )
            
            return MENU_SERVICE_SELECTION
        
        # Extract action from callback data
        if callback_data.startswith(MENU_ACTION):
            action = callback_data[len(MENU_ACTION):]
            
            # Get stored data
            instrument = context.user_data.get("menu_instrument", "")
            market = context.user_data.get("menu_market", "")
            timeframe = context.user_data.get("menu_timeframe", "")
            
            # Construct the symbol
            symbol = instrument
            
            # Process requested action
            if action == "analysis":
                # Show loading message
                await query.edit_message_text(f"Loading technical analysis for {symbol} ({timeframe})...")
                
                try:
                    # Get technical analysis
                    analysis = await self.chart_service.get_technical_analysis(symbol, timeframe)
                    
                    # Check for errors
                    if "error" in analysis:
                        error_msg = analysis.get("message", "Unknown error")
                        keyboard = [[
                            InlineKeyboardButton("🔄 Try Again", callback_data=f"{MENU_ACTION}analysis"),
                            InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK)
                        ]]
                        reply_markup = InlineKeyboardMarkup(keyboard)
                        
                        await query.edit_message_text(
                            text=f"Error getting analysis for {symbol}: {error_msg}",
                            reply_markup=reply_markup
                        )
                        return MENU_ACTION_SELECTION
                    
                    # Format analysis
                    formatted_analysis = format_technical_analysis(analysis)
                    
                    # Create keyboard for options
                    keyboard = [
                        [
                            InlineKeyboardButton("📈 Show Chart", callback_data=f"{MENU_ACTION}chart"),
                            InlineKeyboardButton("🔄 Refresh", callback_data=f"{MENU_ACTION}analysis")
                        ],
                        [
                            InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK),
                            InlineKeyboardButton("🏠 Main Menu", callback_data=f"{MENU_PREFIX}main")
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
                        InlineKeyboardButton("🔄 Try Again", callback_data=f"{MENU_ACTION}analysis"),
                        InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK)
                    ]]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    await query.edit_message_text(
                        text=f"Error getting analysis for {symbol}: {str(e)}",
                        reply_markup=reply_markup
                    )
                
                return MENU_ACTION_SELECTION
                
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
                            InlineKeyboardButton("📊 Technical Analysis", callback_data=f"{MENU_ACTION}analysis"),
                            InlineKeyboardButton("🔄 Refresh Chart", callback_data=f"{MENU_ACTION}chart")
                        ],
                        [
                            InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK),
                            InlineKeyboardButton("🏠 Main Menu", callback_data=f"{MENU_PREFIX}main")
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
                        InlineKeyboardButton("🔄 Try Again", callback_data=f"{MENU_ACTION}chart"),
                        InlineKeyboardButton("⬅️ Back", callback_data=MENU_BACK)
                    ]]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    await query.edit_message_text(
                        text=f"Error generating chart for {symbol}: {str(e)}",
                        reply_markup=reply_markup
                    )
                
                return MENU_ACTION_SELECTION
            
        # Return to main menu if requested
        if callback_data == f"{MENU_PREFIX}main":
            return await MenuFlow.menu_start(update, context)
            
        # Invalid action
        await query.edit_message_text("Invalid selection. Please try again.")
        return ConversationHandler.END
    
    def get_handlers(self) -> List[Any]:
        """
        Get all handlers for the menu flow
        
        Returns:
            List[Any]: List of handlers
        """
        # Create the conversation handler for menu flow
        menu_conv_handler = ConversationHandler(
            entry_points=[
                CommandHandler("menu", self.menu_start),
                CallbackQueryHandler(
                    self.menu_start,
                    pattern=f"^{MENU_PREFIX}main$"
                )
            ],
            states={
                MENU_CATEGORY_SELECTION: [
                    CallbackQueryHandler(
                        self.menu_category_selection,
                        pattern=f"^{MENU_CATEGORY}"
                    )
                ],
                MENU_INSTRUMENT_SELECTION: [
                    CallbackQueryHandler(
                        self.menu_instrument_selection,
                        pattern=f"^{MENU_INSTRUMENT}|{MENU_BACK}$"
                    )
                ],
                MENU_MARKET_SELECTION: [
                    CallbackQueryHandler(
                        self.menu_market_selection,
                        pattern=f"^{MENU_MARKET}|{MENU_BACK}$"
                    )
                ],
                MENU_TIMEFRAME_SELECTION: [
                    CallbackQueryHandler(
                        MenuFlow.menu_timeframe_selection,
                        pattern=f"^{MENU_TIMEFRAME}|{MENU_BACK}$"
                    )
                ],
                MENU_SERVICE_SELECTION: [
                    CallbackQueryHandler(
                        MenuFlow.menu_service_selection,
                        pattern=f"^{MENU_SERVICE}|{MENU_BACK}$"
                    )
                ],
                MENU_ACTION_SELECTION: [
                    CallbackQueryHandler(
                        MenuFlow.menu_action_selection,
                        pattern=f"^{MENU_ACTION}|{MENU_BACK}|{MENU_PREFIX}main$"
                    )
                ]
            },
            fallbacks=[
                # Fallback to end conversation
                CommandHandler("cancel", lambda u, c: ConversationHandler.END)
            ],
            name="menu_flow",
            persistent=False
        )
        
        return [menu_conv_handler] 