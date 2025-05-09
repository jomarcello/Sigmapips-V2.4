#!/usr/bin/env python3
"""
Test script for improved log fetching functionality
"""

import os
import sys
import time
import json
from dotenv import load_dotenv
from webhook_server import fetch_deployment_logs, extract_errors_from_logs

# Load environment variables from .env file
load_dotenv()

# Check if required environment variables are set
if not os.environ.get('RAILWAY_TOKEN'):
    print("RAILWAY_TOKEN environment variable not set. Please set it in your .env file.")
    sys.exit(1)

def test_log_fetching():
    print("\n--- Testing Improved Log Fetching ---")
    
    # Test with the deployment ID that failed previously
    deployment_id = "34d7267b-c301-4ce3-889b-bc01d1305e6b"
    print(f"Fetching logs for deployment: {deployment_id}")
    
    # Measure time taken
    start_time = time.time()
    logs = fetch_deployment_logs(deployment_id)
    end_time = time.time()
    
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    
    if logs:
        print(f"Successfully fetched {len(logs)} characters of logs")
        
        # Extract errors
        errors = extract_errors_from_logs(logs)
        if errors:
            print(f"\nDetected {len(errors)} errors:")
            for i, error in enumerate(errors, 1):
                print(f"{i}. {error}")
        else:
            print("\nNo specific errors detected in logs")
        
        # Save logs to file for inspection
        with open("test_logs_output.txt", "w") as f:
            f.write(logs)
        print(f"\nLogs saved to test_logs_output.txt for inspection")
    else:
        print("Failed to fetch logs")

if __name__ == "__main__":
    test_log_fetching() 