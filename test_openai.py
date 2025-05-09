#!/usr/bin/env python3

import os
import sys
import asyncio
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY environment variable is not set!")
    sys.exit(1)

print(f"Using OpenAI API key: {api_key[:4]}...")

# Initialize the client
client = openai.OpenAI(api_key=api_key)

# Test if OpenAI API works
try:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."}, 
                 {"role": "user", "content": "Say hello!"}]
    )
    print("OpenAI API test successful!")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"OpenAI API test failed with error: {e}")
    sys.exit(1) 