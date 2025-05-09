#!/usr/bin/env python3
"""
Telegram webhook server for SigmaPips Trading Bot.
This server handles incoming webhook requests from Telegram.
"""

import os
import sys
import json
import logging
import traceback
import asyncio
from fastapi import FastAPI, Request, Response, HTTPException, Header
from pydantic import BaseModel
import uvicorn
import signal
from typing import Dict, Any, Optional, List
import time
import threading
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'webhook_server.log'))
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="SigmaPips Trading Bot Webhook Server")

# Get the Telegram bot token from environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
    sys.exit(1)

# Global variable to store the bot instance
bot_instance = None
bot_lock = threading.Lock()

# Track health status
health_status = {
    "startup_time": time.time(),
    "webhook_requests": 0,
    "last_activity": time.time(),
    "errors": 0,
}

class TelegramUpdate(BaseModel):
    update_id: int
    message: Optional[Dict[str, Any]] = None
    edited_message: Optional[Dict[str, Any]] = None
    channel_post: Optional[Dict[str, Any]] = None
    edited_channel_post: Optional[Dict[str, Any]] = None
    inline_query: Optional[Dict[str, Any]] = None
    chosen_inline_result: Optional[Dict[str, Any]] = None
    callback_query: Optional[Dict[str, Any]] = None
    

async def init_bot():
    """Initialize the bot instance"""
    global bot_instance
    
    with bot_lock:
        if bot_instance is not None:
            return bot_instance
            
        logger.info("Initializing bot instance...")
        try:
            # Import here to avoid circular imports
            from trading_bot.services.telegram_service.bot import TelegramService
            from trading_bot.services.database.db import Database
            
            # Initialize database
            db = Database()
            
            # Initialize bot with database
            bot_instance = TelegramService(db=db, bot_token=TELEGRAM_BOT_TOKEN, lazy_init=True)
            await bot_instance.initialize_services()
            
            logger.info("Bot instance initialized successfully")
            return bot_instance
        except Exception as e:
            logger.error(f"Error initializing bot instance: {str(e)}")
            logger.error(traceback.format_exc())
            return None


@app.get("/")
async def root():
    """Root endpoint to verify the server is running"""
    return {"status": "ok", "message": "SigmaPips Trading Bot Webhook Server"}


@app.get("/health")
async def health():
    """Health check endpoint"""
    global health_status
    
    uptime = time.time() - health_status["startup_time"]
    
    return {
        "status": "ok",
        "uptime": uptime,
        "webhook_requests": health_status["webhook_requests"],
        "last_activity": time.time() - health_status["last_activity"],
        "errors": health_status["errors"],
        "bot_initialized": bot_instance is not None
    }


@app.post("/webhook")
async def webhook(update: TelegramUpdate):
    """Handle incoming webhook requests from Telegram"""
    global health_status
    
    # Update health status
    health_status["webhook_requests"] += 1
    health_status["last_activity"] = time.time()
    
    try:
        # Ensure the bot is initialized
        bot = await init_bot()
        if bot is None:
            health_status["errors"] += 1
            raise HTTPException(status_code=500, detail="Bot initialization failed")
        
        # Log the incoming update
        logger.info(f"Received update_id: {update.update_id}")
        
        # Extract message data if available
        message_text = None
        chat_id = None
        message_id = None
        
        if update.message:
            message_text = update.message.get("text", "")
            chat_id = update.message.get("chat", {}).get("id")
            message_id = update.message.get("message_id")
            logger.info(f"Received message: chat_id={chat_id}, message_id={message_id}, text={message_text}")
        
        # Check if the message is a command
        if message_text and message_text.startswith("/"):
            command = message_text.split()[0]
            logger.info(f"Processing command: {command}")
            
            # Special handling for the menu command
            if command == "/menu":
                logger.info("Processing /menu command")
                try:
                    # Send a quick response first
                    await bot.bot.send_message(
                        chat_id=chat_id, 
                        text="Processing menu command...",
                        reply_to_message_id=message_id
                    )
                    
                    # Process the command
                    from telegram import Update
                    from telegram.ext import ContextTypes
                    
                    # Convert dict to Update object
                    telegram_update = Update.de_json(update.dict(), bot.bot)
                    context = ContextTypes.DEFAULT_TYPE(None, None, None, None)
                    
                    # Call the menu_command method
                    await bot.menu_command(telegram_update, context)
                    logger.info("Successfully executed menu_command")
                except Exception as e:
                    logger.error(f"Error processing /menu command: {str(e)}")
                    logger.error(traceback.format_exc())
                    health_status["errors"] += 1
                    
                    # Send error message to user
                    await bot.bot.send_message(
                        chat_id=chat_id,
                        text=f"Error processing menu command: {str(e)}",
                        reply_to_message_id=message_id
                    )
        
        # Return success
        return {"status": "ok"}
    
    except Exception as e:
        logger.error(f"Error handling webhook: {str(e)}")
        logger.error(traceback.format_exc())
        health_status["errors"] += 1
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/set_webhook")
async def set_webhook(url: str = None):
    """Set the webhook URL for the Telegram bot"""
    try:
        # Ensure the bot is initialized
        bot = await init_bot()
        if bot is None:
            raise HTTPException(status_code=500, detail="Bot initialization failed")
        
        # Set the webhook URL
        if url:
            webhook_url = url
        else:
            # Get the host from the environment or use a default
            host = os.getenv("WEBHOOK_HOST", "https://example.com")
            webhook_url = f"{host}/webhook"
        
        # Set the webhook
        result = await bot.bot.set_webhook(url=webhook_url)
        
        if result:
            logger.info(f"Webhook set successfully: {webhook_url}")
            return {"status": "ok", "webhook_url": webhook_url}
        else:
            logger.error(f"Failed to set webhook: {webhook_url}")
            raise HTTPException(status_code=500, detail="Failed to set webhook")
    
    except Exception as e:
        logger.error(f"Error setting webhook: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/delete_webhook")
async def delete_webhook():
    """Delete the webhook for the Telegram bot"""
    try:
        # Ensure the bot is initialized
        bot = await init_bot()
        if bot is None:
            raise HTTPException(status_code=500, detail="Bot initialization failed")
        
        # Delete the webhook
        result = await bot.bot.delete_webhook()
        
        if result:
            logger.info("Webhook deleted successfully")
            return {"status": "ok"}
        else:
            logger.error("Failed to delete webhook")
            raise HTTPException(status_code=500, detail="Failed to delete webhook")
    
    except Exception as e:
        logger.error(f"Error deleting webhook: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize the bot and set the webhook on startup"""
    logger.info("Starting webhook server...")
    
    try:
        # Initialize the bot in the background
        asyncio.create_task(init_bot())
        
        # Log startup
        logger.info("Webhook server started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        logger.error(traceback.format_exc())


def handle_sigterm(signum, frame):
    """Handle SIGTERM signal"""
    logger.info("Received SIGTERM signal, shutting down gracefully")
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    # Check if port is provided in environment
    port = int(os.getenv("PORT", 8000))
    
    # Run the FastAPI app with uvicorn
    logger.info(f"Starting webhook server on port {port}")
    uvicorn.run("webhook_server:app", host="0.0.0.0", port=port, reload=False) 