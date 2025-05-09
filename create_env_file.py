#!/usr/bin/env python3
"""
Script to create a properly formatted .env file from environment variables
"""
import os
import re

# Raw environment variables from command line input with placeholders for sensitive info
raw_vars = """
DEEPSEEK_API_KEY="your_deepseek_api_key"
FORCE_POLLING="true"
FORCE_WEBHOOK="true"
GOOGLE_APPLICATION_CREDENTIALS="/app/google_vision_credentials.json"
GOOGLE_CREDENTIALS_JSON=""
OCR_SPACE_API_KEY="your_ocr_space_api_key"
OPENAI_API_KEY="your_openai_api_key"
PERPLEXITY_API_KEY="your_perplexity_api_key"
PORT="8080"
RAILWAY_PUBLIC_DOMAIN="https://sigmapips-v2-production.up.railway.app"
REDIS_URL="${{Redis-OVJG.REDIS_URL}}"
REDISHOST="${{Redis-OVJG.REDISHOST}}"
REDISPORT="${{Redis-OVJG.REDISPORT}}"
SCRAPINGANT_API_KEY="your_scrapingant_api_key"
STRIPE_LIVE_SECRET_KEY="your_stripe_secret_key"
STRIPE_LIVE_WEBHOOK_SECRET="your_stripe_webhook_secret"
STRIPE_WEBHOOK_SECRET="your_stripe_webhook_secret"
SUPABASE_KEY="your_supabase_key"
SUPABASE_URL="https://utigkgjcyqnrhpndzqhs.supabase.co"
TAVILY_API_KEY="your_tavily_api_key"
TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
TRADINGVIEW_SESSION_ID="your_tradingview_session_id"
TWELVEDATA_API_KEY="your_twelvedata_api_key"
TWOCAPTCHA_API_KEY="your_twocaptcha_api_key"
USE_INVESTING_CALENDAR="true"
USE_SCRAPINGANT="true"
WEBHOOK_PATH="/webhook"
WEBHOOK_URL="sigmapips-v24-production.up.railway.app"
BINANCE_API_KEY="your_binance_api_key"
PYTHONPATH="/app:$PYTHONPATH"
PYTHONUNBUFFERED="1"
RAILWAY_TOKEN="your_railway_token"
"""

# Output file path
env_file = "config/.env"

# Process and format environment variables
formatted_vars = []
for line in raw_vars.strip().split('\n'):
    if not line or line.isspace():
        continue
        
    # Extract key and value
    match = re.match(r'([A-Za-z_0-9]+)=(?:"([^"]*)"|(.*))$', line)
    if match:
        key, value1, value2 = match.groups()
        value = value1 if value1 is not None else value2
        formatted_vars.append(f"{key}={value}")
    else:
        # Skip invalid lines
        print(f"Skipping invalid line: {line}")

# Write to file
os.makedirs(os.path.dirname(env_file), exist_ok=True)
with open(env_file, 'w') as f:
    f.write('\n'.join(formatted_vars))

print(f"Created .env file at {env_file}")
print("IMPORTANT: Replace the placeholder values with your actual API keys and secrets") 