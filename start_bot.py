#!/usr/bin/env python3
"""
Startup script for the SigmaPips Trading Bot.
This script properly configures the Python import path to handle module imports correctly.
"""

import os
import sys
import subprocess
import importlib.util

# Get the absolute path of the project root directory
project_root = os.path.dirname(os.path.abspath(__file__))

# Add the project root to Python path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added {project_root} to Python path")

# Also add the trading_bot subdirectory for direct imports
trading_bot_path = os.path.join(project_root, "trading_bot")
if trading_bot_path not in sys.path:
    sys.path.insert(0, trading_bot_path)
    print(f"Added {trading_bot_path} to Python path")

# Debug paths and imports
print("Python path set to:", sys.path[:5])  # Show first 5 for brevity

# Check if all necessary module files exist
paths_to_check = [
    os.path.join(project_root, "trading_bot", "__init__.py"),
    os.path.join(project_root, "trading_bot", "services", "__init__.py"),
    os.path.join(project_root, "trading_bot", "services", "telegram_service", "__init__.py"),
    os.path.join(project_root, "trading_bot", "services", "telegram_service", "states.py")
]

for path in paths_to_check:
    if not os.path.exists(path):
        print(f"❌ WARNING: Missing critical file {path}")
    else:
        print(f"✅ Found {path}")

# Check if trading_bot module is discoverable
try:
    import trading_bot
    print("✅ 'trading_bot' module found successfully")
except ImportError as e:
    print(f"❌ Error importing trading_bot module: {e}")
    sys.exit(1)

# Now we can import the main module
try:
    print("Starting bot using direct script execution...")
    # Run main.py directly
    try:
        # Try import paths in different ways
        try:
            from trading_bot.services.telegram_service.bot import TelegramService
        except ImportError:
            # Try an alternative import strategy
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "TelegramService", 
                os.path.join(project_root, "trading_bot", "services", "telegram_service", "bot.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            TelegramService = module.TelegramService
            print("✅ Imported TelegramService using alternative method")
    
        import asyncio
        
        # Get necessary imports
        from trading_bot.services.database.db import Database
        from trading_bot.services.payment_service.stripe_service import StripeService
        
        # Get bot token
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not bot_token:
            print("ERROR: TELEGRAM_BOT_TOKEN not found in environment variables")
            sys.exit(1)
            
        # Run the bot
        async def run_bot():
            # Initialize database first
            db = Database()
            # Initialize stripe service with database
            stripe_service = StripeService(db=db)
            # Initialize telegram service
            telegram_service = TelegramService(
                db=db,
                stripe_service=stripe_service,
                bot_token=bot_token
            )
            
            # Start with polling by default
            force_polling = os.environ.get("FORCE_POLLING", "false").lower() == "true"
            
            if force_polling or not getattr(telegram_service, 'webhook_url', None):
                print("Starting in polling mode")
                await telegram_service.application.initialize()
                await telegram_service.initialize_services()
                await telegram_service.application.start()
                await telegram_service.application.updater.start_polling(
                    drop_pending_updates=True,
                    allowed_updates=["message", "callback_query"]
                )
                await telegram_service.application.updater.stop()
                await telegram_service.application.stop()
                await telegram_service.application.shutdown()
            else:
                # Start with webhook
                print(f"Starting with webhook on {telegram_service.webhook_url + telegram_service.webhook_path}")
                await telegram_service.application.initialize()
                await telegram_service.initialize_services()
                await telegram_service.application.start()
                await telegram_service.application.updater.start_webhook(
                    listen="0.0.0.0",
                    port=int(os.environ.get("PORT", "8443")),
                    url_path=telegram_service.webhook_path,
                    webhook_url=telegram_service.webhook_url + telegram_service.webhook_path,
                    drop_pending_updates=True,
                    allowed_updates=["message", "callback_query"]
                )
                await telegram_service.application.updater.stop()
                await telegram_service.application.stop()
                await telegram_service.application.shutdown()
                
        # Run the bot
        asyncio.run(run_bot())
    except Exception as e:
        print(f"❌ Error in main script execution: {e}")
        import traceback
        traceback.print_exc()
        raise
    
except Exception as e:
    print(f"❌ Error starting the bot: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1) 