#!/usr/bin/env python3
"""
Quick test script for testing the improved menu command
"""

import os
import sys
import asyncio
import logging
import traceback
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("quick_menu_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("quick_menu_test")

# Set up the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
logger.info(f"Python path: {sys.path}")

# Load environment variables
load_dotenv()

# Configure higher timeouts for httpx/httpcore
os.environ['HTTPX_TIMEOUT'] = '30.0'  # 30 seconds timeout

# Import directly from the menu_command_fix.py
try:
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from menu_command_fix import (
        improved_menu_command, 
        extract_chat_id, 
        extract_user_id,
        get_bot_instance,
        try_send_menu_gif,
        try_send_text_menu,
        MENU_KEYBOARD,
        WELCOME_MESSAGE
    )
    logger.info("✅ Successfully imported functions from menu_command_fix.py")
except ImportError as e:
    logger.error(f"❌ Error importing from menu_command_fix.py: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Mock telegram update for testing
def create_mock_update(chat_id=123456789, user_id=987654321, username="test_user"):
    """Create a mock update object to simulate a Telegram message"""
    from telegram import Update, Chat, User, Message, Bot
    import datetime
    
    # Create mock user
    user = User(id=user_id, first_name="Test", is_bot=False, username=username)
    
    # Create mock chat
    chat = Chat(id=chat_id, type="private")
    
    # Create mock message with /menu command
    message = Message(
        message_id=1,
        date=datetime.datetime.now(),
        chat=chat,
        from_user=user,
        text="/menu"
    )
    
    # Create the update dictionary with proper structure
    update_dict = {
        'update_id': 1,
        'message': message.to_dict()
    }
    
    # Create update from dict
    update = Update.de_json(update_dict, Bot(token="dummy_token"))
    
    return update

# Create a simple context class that provides what we need
class SimpleContext:
    def __init__(self, bot):
        self.bot = bot

async def run_test():
    """Run the test of the improved menu command"""
    try:
        logger.info("==== TESTING IMPROVED MENU COMMAND ====")
        
        # Get bot token
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not bot_token:
            logger.error("❌ TELEGRAM_BOT_TOKEN environment variable is not set")
            bot_token = input("Enter a valid Telegram bot token for testing: ")
        
        logger.info(f"Using bot token: {bot_token[:4]}...{bot_token[-4:]} to create bot")
        
        # Initialize a minimal bot with proxy support if provided
        from telegram import Bot
        from telegram.request import HTTPXRequest
        
        # Check for proxy settings
        proxy_url = os.getenv("TELEGRAM_PROXY_URL")
        connect_timeout = float(os.getenv("TELEGRAM_CONNECT_TIMEOUT", "20.0"))
        read_timeout = float(os.getenv("TELEGRAM_READ_TIMEOUT", "20.0"))
        
        # Configure request parameters
        request_kwargs = {
            "connect_timeout": connect_timeout,
            "read_timeout": read_timeout
        }
        
        if proxy_url:
            logger.info(f"Using proxy URL: {proxy_url}")
            request_kwargs["proxy_url"] = proxy_url
        
        # Create request object with our parameters
        request = HTTPXRequest(**request_kwargs)
        
        # Create bot with our request object
        bot = Bot(token=bot_token, request=request)
        
        # Test bot connection with higher timeout
        try:
            logger.info("Testing bot connection (with increased timeout)...")
            me = await bot.get_me()
            logger.info(f"✅ Connected to bot: {me.first_name} (@{me.username})")
        except Exception as e:
            logger.error(f"❌ Bot connection failed: {e}")
            logger.error(traceback.format_exc())
            return
        
        # Create a simple context with our bot
        context = SimpleContext(bot)
        
        # Create a mock update
        admin_id = os.getenv("ADMIN_TELEGRAM_ID")
        if admin_id:
            try:
                admin_id = int(admin_id)
                logger.info(f"Using admin ID from environment: {admin_id}")
                update = create_mock_update(chat_id=admin_id, user_id=admin_id)
            except (ValueError, TypeError) as e:
                logger.error(f"❌ Invalid admin ID: {e}")
                update = create_mock_update()
        else:
            update = create_mock_update()
        
        # Log what we're about to test
        logger.info(f"Testing menu command with update: {update}")
        logger.info(f"Chat ID: {extract_chat_id(update)}")
        logger.info(f"User ID: {extract_user_id(update)}")
        
        # Test the command
        logger.info("Calling improved_menu_command...")
        try:
            await improved_menu_command(update, context)
            logger.info("✅ improved_menu_command completed")
        except Exception as e:
            logger.error(f"❌ Error in improved_menu_command: {e}")
            logger.error(traceback.format_exc())
            return
        
        logger.info("==== TEST COMPLETED ====")
        
        # Keep running for a bit to ensure any async operations complete
        logger.info("Waiting 5 seconds for any pending operations...")
        await asyncio.sleep(5)
        
    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        # Set higher default timeouts for all httpx connections
        import httpx
        httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = 30.0
        httpx._config.DEFAULT_TIMEOUT_CONFIG.read = 30.0
        httpx._config.DEFAULT_TIMEOUT_CONFIG.write = 30.0
        httpx._config.DEFAULT_TIMEOUT_CONFIG.pool = 30.0
        
        asyncio.run(run_test())
        logger.info("Test script completed")
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1) 