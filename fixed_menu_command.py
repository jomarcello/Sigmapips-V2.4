"""
Verbeterde implementatie van de menu_command functie voor de Telegram bot.
Na het testen kan deze geïntegreerd worden in de hoofdcode.
"""

import os
import logging
import asyncio
import traceback
from typing import Optional, Union
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ContextTypes, CommandHandler, CallbackContext

# Configureer logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Menu opties en toetsenbord configuratie
MENU_KEYBOARD = [
    [InlineKeyboardButton("📊 Analysis", callback_data="menu_analyse")],
    [InlineKeyboardButton("📈 Signals", callback_data="menu_signals")],
    [InlineKeyboardButton("⚙️ Settings", callback_data="menu_settings")],
    [InlineKeyboardButton("ℹ️ Help", callback_data="menu_help")]
]

WELCOME_MESSAGE = """
📊 <b>SigmaPips AI Trading Bot</b>

Welkom bij de Sigmapips AI Trading Bot. 
Selecteer een optie om te beginnen:
"""

# Verbeterde menu command functie
async def improved_menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE = None) -> None:
    """
    Verbeterde implementatie van het menu commando met uitgebreide foutafhandeling
    en terugvalmechanismen.
    """
    # Begin uitgebreid loggen
    logger.info("===== MENU COMMAND STARTED =====")
    logger.info(f"Update type: {type(update)}")
    if hasattr(update, 'effective_user') and update.effective_user:
        logger.info(f"User: {update.effective_user.id} ({update.effective_user.username})")
    
    # Variabelen voor terugval en tracering
    chat_id = None
    user_id = None
    active_bot = None
    success = False
    error_info = {}
    
    try:
        # Stap 1: Verkrijg chat_id en user_id uit de update
        try:
            chat_id = extract_chat_id(update)
            user_id = extract_user_id(update)
            logger.info(f"Extracted chat_id: {chat_id}, user_id: {user_id}")
            
            if not chat_id:
                logger.error("Could not extract chat_id from update")
                raise ValueError("No valid chat_id available")
        except Exception as e:
            logger.error(f"Error extracting IDs: {e}")
            error_info['id_extraction'] = str(e)
        
        # Stap 2: Verkrijg de bot instantie
        try:
            active_bot = get_bot_instance(update, context)
            logger.info(f"Bot instance retrieved: {active_bot is not None}")
            
            if not active_bot:
                logger.error("No bot instance available")
                raise ValueError("No valid bot instance available")
        except Exception as e:
            logger.error(f"Error getting bot instance: {e}")
            error_info['bot_instance'] = str(e)
            
        # Stap 3: Stel inline keyboard op
        reply_markup = InlineKeyboardMarkup(MENU_KEYBOARD)
        logger.info("Keyboard markup created")
        
        # Stap 4: Probeer het sturen van de GIF bij een menu request voor een betere gebruikerservaring
        if chat_id and active_bot:
            logger.info("Attempting to send GIF")
            success = await try_send_menu_gif(active_bot, chat_id, update, reply_markup)
        
        # Stap 5: Fallback naar text-only als GIF verzenden mislukt
        if not success and chat_id and active_bot:
            logger.info("GIF failed, attempting text-only")
            success = await try_send_text_menu(active_bot, chat_id, update, reply_markup)
        
        # Stap 6: Emergency fallback - zeer eenvoudig bericht
        if not success and chat_id and active_bot:
            logger.info("Text menu failed, attempting emergency fallback")
            simple_keyboard = [
                [InlineKeyboardButton("Menu", callback_data="menu_analyse")]
            ]
            simple_markup = InlineKeyboardMarkup(simple_keyboard)
            
            try:
                await active_bot.send_message(
                    chat_id=chat_id,
                    text="Menu",
                    reply_markup=simple_markup
                )
                logger.info("Emergency fallback successful")
                success = True
            except Exception as e:
                logger.error(f"Emergency fallback failed: {e}")
                error_info['emergency_fallback'] = str(e)
        
        # Stap 7: Log resultaat
        if success:
            logger.info("===== MENU COMMAND SUCCEEDED =====")
        else:
            logger.error(f"===== MENU COMMAND FAILED =====")
            logger.error(f"Errors encountered: {error_info}")
            
    except Exception as e:
        logger.error(f"Critical error in menu_command: {e}")
        logger.error(traceback.format_exc())
        error_info['critical'] = str(e)
    
    logger.info(f"Menu command result: {'SUCCESS' if success else 'FAILED'}")
    return

# Hulpfuncties

def extract_chat_id(update: Update) -> Optional[int]:
    """Extract chat ID from update using various fallback methods"""
    if update is None:
        return None
        
    # Method 1: via effective_chat
    if hasattr(update, 'effective_chat') and update.effective_chat:
        return update.effective_chat.id
        
    # Method 2: via message
    if hasattr(update, 'message') and update.message:
        return update.message.chat_id
        
    # Method 3: via callback_query
    if hasattr(update, 'callback_query') and update.callback_query:
        if hasattr(update.callback_query, 'message') and update.callback_query.message:
            return update.callback_query.message.chat_id
    
    return None

def extract_user_id(update: Update) -> Optional[int]:
    """Extract user ID from update using various fallback methods"""
    if update is None:
        return None
        
    # Method 1: via effective_user
    if hasattr(update, 'effective_user') and update.effective_user:
        return update.effective_user.id
        
    # Method 2: via message.from_user
    if hasattr(update, 'message') and update.message and hasattr(update.message, 'from_user'):
        return update.message.from_user.id
        
    # Method 3: via callback_query.from_user
    if hasattr(update, 'callback_query') and update.callback_query and hasattr(update.callback_query, 'from_user'):
        return update.callback_query.from_user.id
    
    return None

def get_bot_instance(update: Update, context: Optional[ContextTypes.DEFAULT_TYPE]) -> Optional[Bot]:
    """Get a bot instance using various fallback methods"""
    # Method 1: via context.bot
    if context and hasattr(context, 'bot'):
        return context.bot
        
    # Method 2: via update.message.bot
    if update and hasattr(update, 'message') and update.message and hasattr(update.message, 'bot'):
        return update.message.bot
        
    # Method 3: via update.callback_query.message.bot
    if update and hasattr(update, 'callback_query') and update.callback_query and hasattr(update.callback_query, 'message'):
        if update.callback_query.message and hasattr(update.callback_query.message, 'bot'):
            return update.callback_query.message.bot
    
    return None

async def try_send_menu_gif(bot: Bot, chat_id: int, update: Update, reply_markup) -> bool:
    """Try to send menu GIF with multiple fallback methods"""
    try:
        # GIF URL voor het welkomstbericht
        gif_url = "https://media.giphy.com/media/gSzIKNrqtotEYrZv7i/giphy.gif"
        
        # Methode 1: Via update.message.reply_animation
        if update and hasattr(update, 'message') and update.message:
            try:
                await update.message.reply_animation(
                    animation=gif_url,
                    caption=WELCOME_MESSAGE,
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.HTML
                )
                logger.info("Menu GIF sent via update.message.reply_animation")
                return True
            except Exception as e:
                logger.info(f"Failed to send via update.message.reply_animation: {e}")
        
        # Methode 2: Via bot.send_animation
        try:
            await bot.send_animation(
                chat_id=chat_id,
                animation=gif_url,
                caption=WELCOME_MESSAGE,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
            logger.info("Menu GIF sent via bot.send_animation")
            return True
        except Exception as e:
            logger.info(f"Failed to send via bot.send_animation: {e}")
        
        # Methode 3: Probeer met een andere GIF URL
        try:
            alt_gif_url = "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExaHVvdjR6bnY2eWF1NHJqZXRleTg4cDN1MXVxeWNpczRpMm1tMHg2MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/dxld1UBIiGuoh31Fus/giphy.gif"
            await bot.send_animation(
                chat_id=chat_id,
                animation=alt_gif_url,
                caption=WELCOME_MESSAGE,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
            logger.info("Menu GIF sent via alternative URL")
            return True
        except Exception as e:
            logger.info(f"Failed to send via alternative GIF URL: {e}")
        
        return False
    except Exception as e:
        logger.error(f"Error in try_send_menu_gif: {e}")
        return False

async def try_send_text_menu(bot: Bot, chat_id: int, update: Update, reply_markup) -> bool:
    """Try to send text-only menu with multiple fallback methods"""
    try:
        # Methode 1: Via update.message.reply_text
        if update and hasattr(update, 'message') and update.message:
            try:
                await update.message.reply_text(
                    text=WELCOME_MESSAGE,
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.HTML
                )
                logger.info("Text menu sent via update.message.reply_text")
                return True
            except Exception as e:
                logger.info(f"Failed to send via update.message.reply_text: {e}")
        
        # Methode 2: Via bot.send_message
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=WELCOME_MESSAGE,
                reply_markup=reply_markup,
                parse_mode=ParseMode.HTML
            )
            logger.info("Text menu sent via bot.send_message")
            return True
        except Exception as e:
            logger.info(f"Failed to send via bot.send_message: {e}")
            
        # Methode 3: Via bot.send_message met plain text
        try:
            await bot.send_message(
                chat_id=chat_id,
                text="Welcome to SigmaPips AI Trading Bot. Select an option to begin:",
                reply_markup=reply_markup
            )
            logger.info("Simple text menu sent via bot.send_message")
            return True
        except Exception as e:
            logger.info(f"Failed to send simple text menu: {e}")
        
        return False
    except Exception as e:
        logger.error(f"Error in try_send_text_menu: {e}")
        return False

def add_improved_menu_handler(application):
    """Add the improved menu handler to a PTB application"""
    application.add_handler(CommandHandler("menu", improved_menu_command))
    logger.info("Improved menu command handler added to application") 