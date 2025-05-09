#!/usr/bin/env python3
"""
Test script to verify environment variables
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Expected values
expected = {
    'RAILWAY_TOKEN': '***TOKEN_REMOVED***',
    'GITHUB_TOKEN': '***TOKEN_REMOVED***',
    'GITHUB_REPO': 'jomarcello/Sigmapips-V2.4',
    'RAILWAY_PROJECT_ID': '***PROJECT_ID_REMOVED***',
    'LOCAL_REPO_PATH': '/Users/jovannitilborg/Downloads/Sigmapips-V2-5.4-main',
    'WEBHOOK_SERVER_URL': 'https://railwaywebhook-production-8102.up.railway.app',
    'POLL_INTERVAL_SECONDS': '60'
}

# Check environment variables
print("\n--- Environment Variables Check ---")
all_correct = True

for key, expected_value in expected.items():
    actual_value = os.environ.get(key, 'NOT SET')
    match = actual_value == expected_value
    status = "✓ MATCH" if match else "✗ MISMATCH"
    
    if not match:
        all_correct = False
    
    # Mask tokens for security
    display_value = actual_value
    display_expected = expected_value
    if 'TOKEN' in key:
        display_value = actual_value[:8] + '...' if actual_value != 'NOT SET' else 'NOT SET'
        display_expected = expected_value[:8] + '...'
    
    print(f"{key}: {status}")
    print(f"  Expected: {display_expected}")
    print(f"  Actual:   {display_value}")
    if not match:
        print(f"  ** Update needed! **")

# Check .env file
print("\n--- .env File Contents Check ---")
try:
    with open('.env', 'r') as f:
        env_contents = f.read()
        print("Found .env file with the following variables:")
        for line in env_contents.splitlines():
            if line.strip() and not line.startswith('#'):
                key = line.split('=')[0] if '=' in line else line
                print(f"  {key}")
except FileNotFoundError:
    print("No .env file found in the current directory.")

if all_correct:
    print("\n✅ All environment variables match expected values.")
else:
    print("\n⚠️ Some environment variables don't match expected values.")
    print("Please update your .env file with the correct values.") 