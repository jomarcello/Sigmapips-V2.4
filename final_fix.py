#!/usr/bin/env python3

def main():
    # Read the current chart.py file
    with open('trading_bot/services/chart_service/chart.py', 'r') as f:
        content = f.read()
    
    # Look for the method definition and add an except clause before the next method
    method_start = "async def _generate_default_analysis(self, instrument: str, timeframe: str) -> str:"
    next_method = "async def _fetch_crypto_price"
    
    # Split the content at these points
    parts = content.split(method_start)
    if len(parts) < 2:
        print("Could not find the _generate_default_analysis method")
        return
    
    before_method = parts[0]
    method_and_after = parts[1]
    
    method_parts = method_and_after.split(next_method)
    if len(method_parts) < 2:
        print("Could not find the _fetch_crypto_price method")
        return
    
    method_body = method_parts[0]
    after_method = next_method + method_parts[1]
    
    # Add the except clause at the end of the method_body
    fixed_method_body = method_body.rstrip()
    # Check if there's already an except clause
    if "except" not in fixed_method_body:
        fixed_method_body += "\n        except Exception as e:\n"
        fixed_method_body += "            logger.error(f\"Error in default analysis: {str(e)}\")\n"
        fixed_method_body += "            return f\"Unable to generate analysis for {instrument}. Please try again later.\"\n\n"
    
    # Fix the try without except issue
    fixed_before_method = before_method.replace(
        "        try:\n            try:",
        "        # Main provider initialization block\n        try:\n            # First try to load YahooFinance if needed")
    
    # Reassemble the file
    fixed_content = fixed_before_method + method_start + fixed_method_body + after_method
    
    # Write the fixed content to chart.py
    with open('trading_bot/services/chart_service/chart.py', 'w') as f:
        f.write(fixed_content)
    
    print("Fixed chart.py")

if __name__ == "__main__":
    main() 