#!/usr/bin/env python3
"""
Integration script to apply the improved menu command to the TelegramService class
"""

import os
import sys
import logging
import argparse
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("menu_fix_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("menu_fix_integration")

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
logger.info(f"Python path: {sys.path}")

def patch_telegram_service(make_backup=True):
    """
    Patch the TelegramService class with the improved menu_command implementation
    
    Args:
        make_backup (bool): Whether to create a backup of the original file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Locate the necessary files
        menu_fix_path = Path(project_root) / "menu_command_fix.py"
        telegram_service_path = Path(project_root) / "trading_bot" / "services" / "telegram_service" / "bot.py"
        
        if not menu_fix_path.exists():
            logger.error(f"Menu fix file not found at: {menu_fix_path}")
            return False
        
        if not telegram_service_path.exists():
            logger.error(f"TelegramService file not found at: {telegram_service_path}")
            return False
        
        # Read the improved menu command implementation
        logger.info("Reading improved menu command implementation...")
        with open(menu_fix_path, 'r') as f:
            menu_fix_code = f.read()
        
        # Extract the helper functions
        extract_chat_id_func = None
        extract_user_id_func = None
        get_bot_instance_func = None
        try_send_menu_gif_func = None
        try_send_text_menu_func = None
        improved_menu_command_func = None
        
        # Find each function in the code
        import re
        
        # Extract function definitions using regex
        function_pattern = re.compile(r'(def\s+(\w+).*?return.*?)(?=\n\n|$)', re.DOTALL)
        matches = function_pattern.findall(menu_fix_code)
        
        functions = {}
        for func_code, func_name in matches:
            functions[func_name] = func_code
        
        # Sanity check - make sure we found all required functions
        required_functions = [
            'extract_chat_id', 
            'extract_user_id', 
            'get_bot_instance',
            'try_send_menu_gif',
            'try_send_text_menu',
            'improved_menu_command'
        ]
        
        for func_name in required_functions:
            if func_name not in functions:
                logger.error(f"Required function '{func_name}' not found in menu_command_fix.py")
                return False
            else:
                logger.info(f"Found function: {func_name}")
        
        # Make a backup of the original file
        if make_backup:
            backup_path = telegram_service_path.with_suffix(".py.bak")
            logger.info(f"Creating backup at: {backup_path}")
            with open(telegram_service_path, 'r') as src, open(backup_path, 'w') as dst:
                dst.write(src.read())
            logger.info("Backup created successfully")
        
        # Read the TelegramService file
        with open(telegram_service_path, 'r') as f:
            telegram_service_code = f.read()
        
        # Find the menu_command method
        menu_command_pattern = re.compile(r'async\s+def\s+menu_command.*?(?=\n    async def|\n\n\n|$)', re.DOTALL)
        menu_command_match = menu_command_pattern.search(telegram_service_code)
        
        if not menu_command_match:
            logger.error("Could not find menu_command method in TelegramService")
            return False
        
        # Build the replacement code
        helper_functions = "\n\n".join([functions[func] for func in required_functions if func != 'improved_menu_command'])
        
        # Modify the improved_menu_command to match the class method signature
        improved_menu_method = functions['improved_menu_command'].replace(
            "async def improved_menu_command", 
            "async def menu_command"
        )
        
        # Find a good place to insert the helper functions
        # We'll put them right after the class definition
        class_def_pattern = re.compile(r'class\s+TelegramService.*?:')
        class_def_match = class_def_pattern.search(telegram_service_code)
        
        if not class_def_match:
            logger.error("Could not find TelegramService class definition")
            return False
        
        class_def_end = class_def_match.end()
        
        # Split the code at the class definition end
        before_class = telegram_service_code[:class_def_end]
        after_class = telegram_service_code[class_def_end:]
        
        # Insert helper functions
        new_code = before_class + "\n    # Helper functions for improved menu command\n" + \
                  helper_functions.replace("def ", "    def ").replace("\n", "\n    ") + \
                  after_class
        
        # Replace the menu_command method
        new_code = menu_command_pattern.sub(improved_menu_method, new_code)
        
        # Write the modified file
        with open(telegram_service_path, 'w') as f:
            f.write(new_code)
        
        logger.info("Successfully patched TelegramService with improved menu_command")
        return True
    
    except Exception as e:
        logger.error(f"Error patching TelegramService: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate the improved menu command into TelegramService")
    parser.add_argument("--no-backup", action="store_true", help="Don't create a backup of the original file")
    args = parser.parse_args()
    
    make_backup = not args.no_backup
    
    try:
        success = patch_telegram_service(make_backup=make_backup)
        if success:
            print("\n✅ Successfully integrated improved menu command!")
            print("You can now test the bot with the improved menu command using:")
            print("python start_bot_debug.py")
        else:
            print("\n❌ Failed to integrate improved menu command.")
            print("Check menu_fix_integration.log for details.")
    
    except KeyboardInterrupt:
        print("\nScript terminated by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        print(f"\n❌ Unexpected error: {e}") 