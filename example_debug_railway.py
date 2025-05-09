#!/usr/bin/env python3
"""
Example script demonstrating how to use the debugging tools with Railway.
This is a simple walkthrough of the 3-step method described.
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Import our debugging utilities
from trading_bot.utils.debug_logger import DebugLogger, log_input, log_output, log_error, log_process, log_variable
from utils.railway_log_analyzer import RailwayLogAnalyzer

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

def demonstrate_railway_logs(service_name):
    """
    Demonstrate fetching and analyzing Railway logs.
    """
    print("\n=== Railway Log Analysis Demonstration ===\n")
    
    # Initialize the Railway log analyzer
    analyzer = RailwayLogAnalyzer(service_name=service_name)
    
    # Fetch logs from the last hour
    print("Fetching Railway logs from the last hour...")
    log_file = analyzer.fetch_logs(hours=1)
    
    if not log_file:
        print("Failed to fetch Railway logs.")
        return
    
    print(f"Logs fetched and saved to: {log_file}")
    
    # Analyze logs for errors
    print("\nAnalyzing logs for errors...")
    analysis = analyzer.analyze_logs(log_file)
    
    if not analysis:
        print("No errors found in logs.")
        return
    
    print(f"Found {len(analysis)} potential issues:")
    
    # Display the first few issues
    for i, issue in enumerate(analysis[:3], 1):
        print(f"\nIssue {i}:")
        print(f"Error: {issue.get('error_text', 'Unknown error')}")
        print(f"Line: {issue.get('error_line', 'Unknown')}")
        if issue.get('timestamp'):
            print(f"Time: {issue.get('timestamp')}")
    
    # Save analysis for AI processing
    analysis_file = analyzer.save_analysis_for_ai(analysis)
    print(f"\nFull analysis saved to: {analysis_file}")
    print("This file can be used for AI-assisted debugging.")

def main():
    parser = argparse.ArgumentParser(description="Debug Demonstration Script")
    parser.add_argument("--test", choices=["login", "signal", "railway"], required=True,
                        help="Test to run (login, signal, or railway)")
    parser.add_argument("--service", default="sigmapips-bot",
                        help="Railway service name for log analysis")
    parser.add_argument("--error", action="store_true",
                        help="Trigger an error scenario for demonstration")
    
    args = parser.parse_args()
    
    # Initialize the logger
    logger = DebugLogger(log_name="example_debug", log_dir="logs")
    
    print("\n=== AI-Powered Debugging Demonstration ===\n")
    
    if args.test == "login":
        print("Testing user login flow with debug logging...")
        try:
            username = "test_error" if args.error else "test_user"
            result = simulate_user_login(username, "test_password")
            print(f"Login result: {result}")
        except Exception as e:
            print(f"Error during login: {e}")
        print("\nCheck logs/example_debug.log for detailed debugging information.")
    
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
        print("\nCheck logs/example_debug.log for detailed debugging information.")
    
    elif args.test == "railway":
        demonstrate_railway_logs(args.service)

if __name__ == "__main__":
    main() 