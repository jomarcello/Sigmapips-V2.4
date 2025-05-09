#!/usr/bin/env python3
"""
Standalone example script demonstrating the debugging workflow.
This version doesn't rely on the trading_bot module to avoid import issues.
"""

import os
import sys
import time
import json
import argparse
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import traceback

# Create the logs directory
os.makedirs("logs", exist_ok=True)
os.makedirs("railway_logs", exist_ok=True)

# Simple standalone debug logger implementation
class DebugLogger:
    def __init__(self, log_name="debug", log_dir="logs", max_size_mb=10, backup_count=5):
        self.log_name = log_name
        self.log_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up the logger
        self.logger = logging.getLogger(f"debug_{log_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create rotating file handler
        log_file = os.path.join(self.log_dir, f"{log_name}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

# Create a default instance
default_logger = DebugLogger()

# Helper functions
def log_input(input_data, source="unknown"):
    try:
        if isinstance(input_data, (dict, list)):
            input_str = json.dumps(input_data, default=str)
        else:
            input_str = str(input_data)
        
        default_logger.logger.info(f"INPUT [{source}]: {input_str}")
    except Exception as e:
        default_logger.logger.error(f"Failed to log input: {str(e)}")

def log_output(output_data, destination="unknown"):
    try:
        if isinstance(output_data, (dict, list)):
            output_str = json.dumps(output_data, default=str)
        else:
            output_str = str(output_data)
        
        default_logger.logger.info(f"OUTPUT [{destination}]: {output_str}")
    except Exception as e:
        default_logger.logger.error(f"Failed to log output: {str(e)}")

def log_error(error, context=None):
    try:
        error_msg = f"ERROR: {str(error)}"
        if context:
            if isinstance(context, dict):
                context_str = json.dumps(context, default=str)
            else:
                context_str = str(context)
            error_msg += f" | Context: {context_str}"
        
        default_logger.logger.error(error_msg)
        default_logger.logger.error(f"Traceback: {traceback.format_exc()}")
    except Exception as e:
        default_logger.logger.error(f"Failed to log error: {str(e)}")

def log_process(process_name, data=None, status="running"):
    try:
        msg = f"PROCESS [{process_name}] Status: {status}"
        if data:
            if isinstance(data, dict):
                data_str = json.dumps(data, default=str)
            else:
                data_str = str(data)
            msg += f" | Data: {data_str}"
        
        default_logger.logger.info(msg)
    except Exception as e:
        default_logger.logger.error(f"Failed to log process: {str(e)}")

def log_variable(var_name, var_value):
    try:
        if isinstance(var_value, (dict, list)):
            value_str = json.dumps(var_value, default=str)
        else:
            value_str = str(var_value)
        
        default_logger.logger.debug(f"VARIABLE [{var_name}]: {value_str}")
    except Exception as e:
        default_logger.logger.error(f"Failed to log variable: {str(e)}")

# Example functions with debug logging
def simulate_user_login(username, password):
    """
    Simulate a user login process with intentional errors for demonstration.
    """
    log_input({"username": username, "password": "********"}, source="simulate_user_login")
    log_process("user_login", {"username": username}, "started")
    
    try:
        # Simulate API call to authenticate
        print(f"Authenticating user: {username}")
        log_variable("auth_start_time", datetime.now().isoformat())
        
        # Simulate some processing time
        time.sleep(1)
        
        # Simulate a random error sometimes
        if username == "test_error":
            # This will cause an error for demonstration
            result = {"user": {"id": None, "name": username}["nonexistent_key"]}
            log_output(result, "user_authentication")
            return result
        
        # Normal successful flow
        result = {"success": True, "user_id": 12345, "username": username}
        log_output(result, "user_authentication")
        log_process("user_login", {"username": username}, "completed")
        return result
        
    except Exception as e:
        log_error(e, {"username": username, "function": "simulate_user_login"})
        log_process("user_login", {"username": username}, "failed")
        raise

def process_trading_signal(signal_id, signal_data=None):
    """
    Simulate processing a trading signal.
    """
    log_input({"signal_id": signal_id, "signal_data": signal_data}, source="process_trading_signal")
    log_process("signal_processing", {"signal_id": signal_id}, "started")
    
    try:
        # Simulate validation error
        if not signal_data:
            error_msg = "Signal data is required"
            log_error(error_msg, {"signal_id": signal_id})
            log_process("signal_processing", {"signal_id": signal_id}, "failed")
            return {"success": False, "error": error_msg}
        
        # Simulate processing
        print(f"Processing signal: {signal_id}")
        log_variable("signal_type", signal_data.get("type", "unknown"))
        
        # Simulate API call to process the signal
        time.sleep(1)
        
        # Intentional error for demonstration
        if signal_id == "error_signal":
            # This will cause a KeyError
            pair = signal_data["trading_pair"]["symbol"]
            log_variable("pair", pair)
        
        # Process the signal
        result = {
            "success": True,
            "signal_id": signal_id,
            "processed_at": datetime.now().isoformat(),
            "action": "BUY" if signal_id.startswith("buy") else "SELL"
        }
        
        log_output(result, "signal_processing")
        log_process("signal_processing", {"signal_id": signal_id}, "completed")
        return result
        
    except Exception as e:
        log_error(e, {"signal_id": signal_id, "function": "process_trading_signal"})
        log_process("signal_processing", {"signal_id": signal_id}, "failed")
        raise

def simulate_railway_logs():
    """
    Simulate Railway logs analysis since we might not have Railway CLI access.
    """
    print("\n=== Simulated Railway Log Analysis ===\n")
    
    # Create a sample log file
    log_file = os.path.join("railway_logs", f"simulated_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    # Write some sample logs
    with open(log_file, "w") as f:
        f.write("2023-05-09T10:15:23Z - INFO - Application started\n")
        f.write("2023-05-09T10:15:24Z - INFO - Connected to database\n")
        f.write("2023-05-09T10:15:30Z - INFO - User login: user123\n")
        f.write("2023-05-09T10:15:35Z - ERROR - KeyError: 'auth_token' in authenticate_user at /app/trading_bot/auth.py:45\n")
        f.write("2023-05-09T10:15:35Z - ERROR - Traceback: File \"/app/trading_bot/auth.py\", line 45, in authenticate_user\n")
        f.write("    token = user_data['auth_token']\n")
        f.write("KeyError: 'auth_token'\n")
        f.write("2023-05-09T10:15:40Z - INFO - Signal received: BTC_LONG\n")
        f.write("2023-05-09T10:15:45Z - ERROR - TypeError: 'NoneType' object is not subscriptable in process_signal at /app/trading_bot/signals.py:78\n")
        f.write("2023-05-09T10:15:45Z - ERROR - Traceback: File \"/app/trading_bot/signals.py\", line 78, in process_signal\n")
        f.write("    price = signal_data['price']['value']\n")
        f.write("TypeError: 'NoneType' object is not subscriptable\n")
    
    print(f"Created simulated Railway logs at: {log_file}")
    
    # Analyze the logs
    print("\nAnalyzing logs for errors...")
    
    # Simple error detection
    errors = []
    with open(log_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "ERROR" in line:
                # Get context (5 lines before and after)
                start = max(0, i - 5)
                end = min(len(lines), i + 6)
                context = ''.join(lines[start:end])
                
                # Extract timestamp
                timestamp = line.split(" - ")[0] if " - " in line else None
                
                errors.append({
                    "error_line": i + 1,
                    "error_text": line.strip(),
                    "context": context,
                    "timestamp": timestamp
                })
    
    if not errors:
        print("No errors found in logs.")
        return
    
    print(f"Found {len(errors)} potential issues:")
    
    # Display the issues
    for i, issue in enumerate(errors, 1):
        print(f"\nIssue {i}:")
        print(f"Error: {issue['error_text']}")
        print(f"Line: {issue['error_line']}")
        if issue['timestamp']:
            print(f"Time: {issue['timestamp']}")
    
    # Save analysis for AI processing
    analysis_file = os.path.join("railway_logs", f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(analysis_file, "w") as f:
        json.dump(errors, f, indent=2)
    
    print(f"\nFull analysis saved to: {analysis_file}")
    print("This file can be used for AI-assisted debugging.")
    
    # Generate recommendations
    print("\nAI-Generated Recommendations:")
    print("1. Check auth.py line 45: Missing 'auth_token' key in user_data dictionary")
    print("2. Check signals.py line 78: signal_data['price'] is None, add validation")
    print("3. Suggested files to inspect: auth.py, signals.py")
    print("4. Suggested functions to debug: authenticate_user, process_signal")

def main():
    parser = argparse.ArgumentParser(description="Debug Demonstration Script")
    parser.add_argument("--test", choices=["login", "signal", "railway"], required=True,
                        help="Test to run (login, signal, or railway)")
    parser.add_argument("--error", action="store_true",
                        help="Trigger an error scenario for demonstration")
    
    args = parser.parse_args()
    
    print("\n=== AI-Powered Debugging Demonstration ===\n")
    
    if args.test == "login":
        print("Testing user login flow with debug logging...")
        try:
            username = "test_error" if args.error else "test_user"
            result = simulate_user_login(username, "test_password")
            print(f"Login result: {result}")
        except Exception as e:
            print(f"Error during login: {e}")
        print("\nCheck logs/debug.log for detailed debugging information.")
    
    elif args.test == "signal":
        print("Testing signal processing with debug logging...")
        try:
            signal_id = "error_signal" if args.error else "buy_btc_signal"
            signal_data = {
                "type": "market",
                "direction": "long",
                "price": 50000.0
            }
            
            # If we want to trigger a missing data error
            if args.error:
                signal_data["trading_pair"] = {"name": "BTC/USD"}  # Missing 'symbol' key
            else:
                signal_data["trading_pair"] = {"symbol": "BTC/USD", "name": "Bitcoin/US Dollar"}
                
            result = process_trading_signal(signal_id, signal_data)
            print(f"Signal processing result: {result}")
        except Exception as e:
            print(f"Error during signal processing: {e}")
        print("\nCheck logs/debug.log for detailed debugging information.")
    
    elif args.test == "railway":
        simulate_railway_logs()

if __name__ == "__main__":
    main() 