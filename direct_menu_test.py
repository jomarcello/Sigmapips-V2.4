#!/usr/bin/env python3
"""
Direct test for the menu command - sending an actual message to a real admin user
This avoids all the mocking complexity
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
        logging.FileHandler("direct_menu_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("direct_menu_test")

# Load environment variables
load_dotenv()

# Default admin ID for testing - CHANGE THIS TO YOUR OWN TELEGRAM ID
DEFAULT_ADMIN_ID = 123456789  

async def direct_test():
    """Send a direct message to admin to test menu command functionality"""
    try:
        # Get bot token and admin ID
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        admin_id = os.getenv("ADMIN_TELEGRAM_ID", "2004519703")
        
        if not bot_token:
            logger.error("❌ TELEGRAM_BOT_TOKEN not set in environment")
            return False
        
        # Try to convert to integer
        try:
            admin_id = int(admin_id)
        except ValueError:
            logger.error(f"Invalid admin ID: {admin_id}")
            admin_id = 2004519703  # Fallback to a default ID
        
        logger.info(f"Using bot token: {bot_token[:4]}...{bot_token[-4:]}")
        logger.info(f"Sending to admin ID: {admin_id}")
        
        # Import from menu_command_fix
        from menu_command_fix import WELCOME_MESSAGE, MENU_KEYBOARD
        
        # Initialize a bot
        from telegram import Bot, InlineKeyboardMarkup
        from telegram.constants import ParseMode
        from telegram.request import HTTPXRequest
        
        # Configure higher timeout
        request = HTTPXRequest(
            connect_timeout=20.0,
            read_timeout=20.0
        )
        
        # Create bot with our request object
        bot = Bot(token=bot_token, request=request)
        
        # Test connection
        try:
            logger.info("Testing bot connection...")
            me = await bot.get_me()
            logger.info(f"✅ Connected to bot: {me.first_name} (@{me.username})")
        except Exception as e:
            logger.error(f"❌ Bot connection failed: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # Create keyboard markup
        keyboard_markup = InlineKeyboardMarkup(MENU_KEYBOARD)
        
        # First, try sending a regular text message
        try:
            logger.info("Sending text message...")
            message = await bot.send_message(
                chat_id=admin_id,
                text="🧪 TEST: This is a test of the menu command"
            )
            logger.info(f"✅ Text message sent successfully (message ID: {message.message_id})")
        except Exception as e:
            logger.error(f"❌ Error sending text message: {e}")
            logger.error(traceback.format_exc())
            return False
        
        # Then, try sending GIF with buttons
        try:
            logger.info("Sending GIF with menu buttons...")
            gif_url = "https://media.giphy.com/media/gSzIKNrqtotEYrZv7i/giphy.gif"
            animation = await bot.send_animation(
                chat_id=admin_id,
                animation=gif_url,
                caption=WELCOME_MESSAGE,
                reply_markup=keyboard_markup,
                parse_mode=ParseMode.HTML
            )
            logger.info(f"✅ GIF sent successfully (message ID: {animation.message_id})")
        except Exception as e:
            logger.error(f"❌ Error sending GIF: {e}")
            logger.error(traceback.format_exc())
            
            # Try alternative GIF URL
            try:
                logger.info("Trying alternative GIF URL...")
                alt_gif_url = "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExaHVvdjR6bnY2eWF1NHJqZXRleTg4cDN1MXVxeWNpczRpMm1tMHg2MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/dxld1UBIiGuoh31Fus/giphy.gif"
                animation = await bot.send_animation(
                    chat_id=admin_id,
                    animation=alt_gif_url,
                    caption=WELCOME_MESSAGE,
                    reply_markup=keyboard_markup,
                    parse_mode=ParseMode.HTML
                )
                logger.info(f"✅ Alternative GIF sent successfully (message ID: {animation.message_id})")
            except Exception as alt_e:
                logger.error(f"❌ Error sending alternative GIF: {alt_e}")
                logger.error(traceback.format_exc())
                
                # Fall back to text-only menu
                try:
                    logger.info("Falling back to text-only menu...")
                    message = await bot.send_message(
                        chat_id=admin_id,
                        text=WELCOME_MESSAGE,
                        reply_markup=keyboard_markup,
                        parse_mode=ParseMode.HTML
                    )
                    logger.info(f"✅ Text menu sent successfully (message ID: {message.message_id})")
                except Exception as text_e:
                    logger.error(f"❌ Error sending text menu: {text_e}")
                    logger.error(traceback.format_exc())
                    
                    # Try plain text as last resort
                    try:
                        logger.info("Trying plain text as last resort...")
                        message = await bot.send_message(
                            chat_id=admin_id,
                            text="SigmaPips Menu - Test of fallback mechanism",
                            reply_markup=keyboard_markup
                        )
                        logger.info(f"✅ Plain text menu sent successfully (message ID: {message.message_id})")
                    except Exception as plain_e:
                        logger.error(f"❌ All menu sending methods failed: {plain_e}")
                        logger.error(traceback.format_exc())
                        return False
        
        logger.info("✅ Menu command test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    try:
        # Allow command-line override of admin ID
        if len(sys.argv) > 1:
            try:
                DEFAULT_ADMIN_ID = int(sys.argv[1])
                print(f"Using admin ID from command line: {DEFAULT_ADMIN_ID}")
            except ValueError:
                print(f"Invalid admin ID argument: {sys.argv[1]}")
                sys.exit(1)
        
        # Set higher default timeouts for all httpx connections
        import httpx
        httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = 30.0
        httpx._config.DEFAULT_TIMEOUT_CONFIG.read = 30.0
        httpx._config.DEFAULT_TIMEOUT_CONFIG.write = 30.0
        httpx._config.DEFAULT_TIMEOUT_CONFIG.pool = 30.0
        
        result = asyncio.run(direct_test())
        
        if result:
            logger.info("All tests completed successfully!")
            sys.exit(0)
        else:
            logger.error("Tests failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(2)
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(3) 