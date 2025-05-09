#!/usr/bin/env python3
"""
Test script to simulate a failing Railway deployment and test the Cursor bridge
"""

import os
import sys
import json
import tempfile
from dotenv import load_dotenv

# Import the cursor bridge functionality
from cursor_bridge_enhanced import create_prompt_file, extract_errors_from_logs, launch_cursor_with_prompt

# Load environment variables
load_dotenv()

# Check if required environment variables are set
if not os.environ.get('RAILWAY_TOKEN'):
    print("RAILWAY_TOKEN environment variable not set.")
    sys.exit(1)

def test_cursor_bridge():
    """Test the Cursor bridge with a simulated failing deployment"""
    print("\n--- Testing Cursor Bridge with Simulated Deployment ---")
    
    # Create a test deployment ID
    deployment_id = "test-deployment-123"
    service_name = "Sigmapips-V2.4-Test"
    
    # Create simulated logs with common errors
    logs = """
Starting deployment...
Installing dependencies...
ERROR: Cannot find module 'pydantic'
ModuleNotFoundError: No module named 'pydantic'
npm ERR! code 1
npm ERR! path /app
npm ERR! command failed
npm ERR! 

Error importing: cannot import name 'ChartService' from partially initialized module 'trading_bot.services.chart_service.chart' (most likely due to a circular import) (/app/trading_bot/services/chart_service/chart.py)
Python path: ['/app', '', '/app', '/usr/local/lib/python310.zip', '/usr/local/lib/python3.10', '/usr/local/lib/python3.10/lib-dynload', '/usr/local/lib/python3.10/site-packages']

The process '/usr/bin/python3' failed with exit code 1
Waiting for health check...
Health check failed after 10 attempts
Deployment failed.
"""

    # Create test error log file
    test_error_log_path = os.path.join(os.getcwd(), "test-deployment/error_log.txt")
    os.makedirs(os.path.dirname(test_error_log_path), exist_ok=True)
    
    with open(test_error_log_path, "w") as f:
        f.write(logs)
    
    print(f"Created test error log at: {test_error_log_path}")
    
    # Extract errors from logs
    errors = extract_errors_from_logs(logs)
    
    if errors:
        print(f"\nDetected {len(errors)} errors in logs:")
        for i, error in enumerate(errors, 1):
            print(f"{i}. {error}")
    
    # Create prompt file
    prompt_file = create_prompt_file(deployment_id, logs, errors)
    print(f"\nCreated prompt file at: {prompt_file}")
    
    # Ask if user wants to open Cursor
    repo_path = os.environ.get('LOCAL_REPO_PATH')
    if not repo_path:
        repo_path = os.path.dirname(os.getcwd())
    
    open_cursor = input("\nDo you want to open Cursor with this prompt? (y/n): ").lower().strip() == 'y'
    
    if open_cursor:
        print(f"Launching Cursor with repo path: {repo_path}")
        launch_cursor_with_prompt(prompt_file, repo_path)
        print("Cursor launched successfully!")
    else:
        print("Skipping Cursor launch.")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_cursor_bridge() 