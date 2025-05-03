#!/usr/bin/env python3

def main():
    # Read the original file
    with open("trading_bot/services/chart_service/chart.py", "r") as f:
        content = f.read()
    
    # Fix #1: The try at line ~657-660 without except
    # Strategy: Replace the try without except with a properly structured try/except block
    fixed_content = content.replace(
        """        try:
            try:""",
        """        # Main provider initialization block
        try:
            # Load providers
            """
    )
    
    # Fix #2: Add missing except for the outer try block at the end of the method
    # Find the closing marker for the method (the next async def after _generate_default_analysis)
    method_start = "async def _generate_default_analysis(self, instrument: str, timeframe: str) -> str:"
    next_method_start = "async def _fetch_crypto_price"
    
    if method_start in fixed_content and next_method_start in fixed_content:
        method_section = fixed_content.split(method_start)[1].split(next_method_start)[0]
        
        # Check if the last try block has a matching except
        if method_section.count("try:") > method_section.count("except"):
            # We need to add an except clause at the end of the method
            section_with_except = method_section.rstrip() + "\n        except Exception as e:\n            logger.error(f\"Fallback analysis error: {str(e)}\")\n            return f\"Error generating analysis for {instrument}.\"\n\n    "
            
            # Replace the original method section
            fixed_content = fixed_content.replace(method_section, section_with_except)
    
    # Write the fixed content back
    with open("trading_bot/services/chart_service/chart.py", "w") as f:
        f.write(fixed_content)
    
    print("Fixed try-except blocks in chart.py")

if __name__ == "__main__":
    main() 