#!/usr/bin/env python3
"""
Clean start script for the trading bot.
This script ensures proper cleanup before starting the bot.
"""

import os
import sys
import logging
import asyncio
import subprocess
from telegram import Bot
from telegram.error import TelegramError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get bot token from environment or ask user to provide it
def get_bot_token():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        print("\033[91mERROR: TELEGRAM_BOT_TOKEN not found in environment variables.\033[0m")
        print("Please enter your Telegram bot token (obtained from BotFather):")
        token = input("> ")
        if token:
            # Save to environment for this session
            os.environ["TELEGRAM_BOT_TOKEN"] = token
            # Try to add to .env file for future sessions
            try:
                with open(".env", "a+") as f:
                    f.seek(0)
                    content = f.read()
                    if "TELEGRAM_BOT_TOKEN" not in content:
                        f.write(f"\nTELEGRAM_BOT_TOKEN={token}\n")
            except Exception as e:
                logger.warning(f"Could not save token to .env file: {e}")
        else:
            logger.error("No token provided. Cannot continue.")
            sys.exit(1)
    return token

BOT_TOKEN = get_bot_token()

async def cleanup_before_start():
    """Clean up any existing bot instances and webhooks"""
    try:
        # Step 1: Stop any existing bot processes
        logger.info("Checking for existing bot instances...")
        subprocess.run(["python3", "stop_existing_bots.py"], check=True)
        
        # Step 2: Delete webhook with drop_pending_updates
        logger.info("Attempting to clear Telegram API sessions...")
        bot = Bot(token=BOT_TOKEN)
        
        # Try to get webhook info first
        try:
            webhook_info = await bot.get_webhook_info()
            if webhook_info.url:
                logger.info(f"Found existing webhook at: {webhook_info.url}")
                if webhook_info.pending_update_count > 0:
                    logger.info(f"Webhook has {webhook_info.pending_update_count} pending updates that will be dropped")
        except Exception as e:
            logger.error(f"Error checking webhook: {e}")
        
        # Try to delete webhook with multiple retries
        logger.info("Sending dummy getUpdates request to clear sessions...")
        try:
            await bot.get_updates(timeout=1, offset=-1, limit=1)
        except Exception as e:
            logger.info(f"Expected exception from getUpdates: {e}")
            
        # Step 3: Check webhook status
        try:
            webhook_info = await bot.get_webhook_info()
            if webhook_info.url:
                logger.warning(f"Webhook is still set: {webhook_info.url}")
            else:
                logger.info("Webhook removed successfully, ready for polling")
        except Exception as e:
            logger.error(f"Error checking webhook status: {e}")
            
        # Return success status
        return True
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return False

async def main():
    """Main function to clean up and start the bot"""
    try:
        # Perform cleanup
        cleanup_successful = await cleanup_before_start()
        
        if cleanup_successful:
            logger.info("Done. It's now safe to start a new bot instance.")
            # Start the main script with proper environmental variables
            os.environ["FORCE_POLLING"] = "true"
            # Change directory to the root to ensure imports work correctly
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            # Use module import instead of direct script execution
            subprocess.run(["python3", "trading_bot/main.py"], check=True)
        else:
            logger.error("Cleanup failed, not starting bot")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 
