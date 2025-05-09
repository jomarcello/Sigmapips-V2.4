"""
Configuration settings for the Trading Bot
"""

import os
from dotenv import load_dotenv
import re
load_dotenv()
# Browser settings
DISABLE_BROWSER = os.environ.get("DISABLE_BROWSER", "false").lower() == "true"

# Cache settings
CACHE_ENABLED = True
CACHE_TTL = 300  # 5 minutes in seconds

# API settings
API_TIMEOUT = 30  # seconds
API_RETRY_COUNT = 3

# OpenAI API Configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Validate OpenAI API key format
def validate_openai_key(key):
    """Validate OpenAI API key format"""
    # Accept both standard OpenAI API keys (sk-...) and project-based keys (sk-proj-...)
    # Also allow + in the key for project-based keys
    pattern = r"^sk-(?:proj-)?[a-zA-Z0-9_\-+]+$"
    if not re.match(pattern, key):
        raise ValueError("Invalid OpenAI API key format")
        
validate_openai_key(OPENAI_API_KEY)

# Chart settings
DEFAULT_CHART_TIMEFRAME = "1h"

# Debug print
print(f"Loaded OPENAI_API_KEY: {OPENAI_API_KEY[:4]}...")
