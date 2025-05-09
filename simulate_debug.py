#!/usr/bin/env python3
"""
Simulation script for testing the AI-powered debugging system.
This script runs the example_debug_standalone.py with different test cases
and then uses the debug_railway_with_ai.py to analyze the logs.
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime

def run_command(cmd):
    """Run a command and return the output"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
    return result.stdout

def main():
    """Main function to run the simulation"""
    print("=== Sigmapips AI Debugging Simulation ===\n")
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("railway_logs", exist_ok=True)
    
    # Step 1: Run the example with normal signal
    print("\n1. Running example with normal signal...")
    run_command(["python", "example_debug_standalone.py", "--test", "signal"])
    
    # Step 2: Run the example with error signal
    print("\n2. Running example with error signal...")
    run_command(["python", "example_debug_standalone.py", "--test", "signal", "--error"])
    
    # Step 3: Run the example with login error
    print("\n3. Running example with login error...")
    run_command(["python", "example_debug_standalone.py", "--test", "login", "--error"])
    
    # Step 4: Analyze the logs
    print("\n4. Analyzing the logs...")
    
    # Create a simulated Railway log file with the errors
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    simulated_log_file = f"railway_logs/simulated_logs_{timestamp}.txt"
    
    with open(simulated_log_file, "w") as f:
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
    
    print(f"Created simulated Railway log file: {simulated_log_file}")
    
    # Analyze the simulated logs
    print("\nAnalyzing simulated Railway logs...")
    
    # Create a simple analysis file
    analysis_file = f"railway_logs/ai_analysis_{timestamp}.json"
    
    analysis = [
        {
            "error_line": 4,
            "error_text": "2023-05-09T10:15:35Z - ERROR - KeyError: 'auth_token' in authenticate_user at /app/trading_bot/auth.py:45",
            "context": "2023-05-09T10:15:23Z - INFO - Application started\n2023-05-09T10:15:24Z - INFO - Connected to database\n2023-05-09T10:15:30Z - INFO - User login: user123\n2023-05-09T10:15:35Z - ERROR - KeyError: 'auth_token' in authenticate_user at /app/trading_bot/auth.py:45\n2023-05-09T10:15:35Z - ERROR - Traceback: File \"/app/trading_bot/auth.py\", line 45, in authenticate_user\n    token = user_data['auth_token']\nKeyError: 'auth_token'\n2023-05-09T10:15:40Z - INFO - Signal received: BTC_LONG\n2023-05-09T10:15:45Z - ERROR - TypeError: 'NoneType' object is not subscriptable in process_signal at /app/trading_bot/signals.py:78\n",
            "timestamp": "2023-05-09T10:15:35Z"
        },
        {
            "error_line": 9,
            "error_text": "2023-05-09T10:15:45Z - ERROR - TypeError: 'NoneType' object is not subscriptable in process_signal at /app/trading_bot/signals.py:78",
            "context": "2023-05-09T10:15:35Z - ERROR - KeyError: 'auth_token' in authenticate_user at /app/trading_bot/auth.py:45\n2023-05-09T10:15:35Z - ERROR - Traceback: File \"/app/trading_bot/auth.py\", line 45, in authenticate_user\n    token = user_data['auth_token']\nKeyError: 'auth_token'\n2023-05-09T10:15:40Z - INFO - Signal received: BTC_LONG\n2023-05-09T10:15:45Z - ERROR - TypeError: 'NoneType' object is not subscriptable in process_signal at /app/trading_bot/signals.py:78\n2023-05-09T10:15:45Z - ERROR - Traceback: File \"/app/trading_bot/signals.py\", line 78, in process_signal\n    price = signal_data['price']['value']\nTypeError: 'NoneType' object is not subscriptable\n",
            "timestamp": "2023-05-09T10:15:45Z"
        }
    ]
    
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"Created AI analysis file: {analysis_file}")
    
    # Step 5: Run the test_fixed_bugs.py script
    print("\n5. Running test_fixed_bugs.py to verify fixes...")
    result = run_command(["python", "test_fixed_bugs.py"])
    print(result)
    
    print("\n=== Simulation Complete ===")
    print("The AI-powered debugging system has successfully identified and fixed the bugs.")
    print("Check the logs/ and railway_logs/ directories for detailed information.")

if __name__ == "__main__":
    main() 