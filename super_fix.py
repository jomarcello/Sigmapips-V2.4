#!/usr/bin/env python3

# This script reads the original chart.py, fixes all issues, and writes a new file

with open('trading_bot/services/chart_service/chart.py.bak', 'r') as f:
    content = f.read()

# Fix #1: The try without except issue
fixed_content = content.replace(
    "        try:\n            try:",
    "        # Main provider initialization\n        try:\n            # First try to load YahooFinance"
)

# Fix #2: Add a missing except clause to the _generate_default_analysis method
method_start_idx = fixed_content.find("async def _generate_default_analysis")
next_method_idx = fixed_content.find("async def _fetch_crypto_price")

if method_start_idx != -1 and next_method_idx != -1:
    method_content = fixed_content[method_start_idx:next_method_idx]
    
    # Check if the method has a try but no except
    if method_content.count("try:") > method_content.count("except"):
        # Find the end of the method content (right before the next method)
        # Check if there's already proper indentation at this point
        method_end = next_method_idx - 1
        while method_end > method_start_idx and fixed_content[method_end].isspace():
            method_end -= 1
        
        # Add the except clause
        except_clause = "\n        except Exception as e:\n"
        except_clause += "            logger.error(f\"Error in default analysis: {str(e)}\")\n"
        except_clause += "            return f\"Unable to generate analysis for {instrument}. Please try again later.\"\n\n    "
        
        # Insert the except clause
        fixed_content = fixed_content[:method_end+1] + except_clause + fixed_content[method_end+1:]

# Fix #3: Fix indentation for the method definition if needed
fixed_content = fixed_content.replace(
    "async def _generate_default_analysis",
    "    async def _generate_default_analysis"
)

# Write the fixed content to the chart.py file
with open('trading_bot/services/chart_service/chart.py', 'w') as f:
    f.write(fixed_content)

print("Fixed chart.py file with all issues corrected") 