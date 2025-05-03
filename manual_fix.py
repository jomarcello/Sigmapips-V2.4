#!/usr/bin/env python3

"""
Script to fix the specific syntax error in chart.py (missing colon after try)
"""

import os

def main():
    # Read current file content
    with open("trading_bot/services/chart_service/chart.py", "r") as f:
        lines = f.readlines()
    
    # Fix 1: The try without except issue (around line 657)
    # Find the problematic section
    start_idx = None
    for i, line in enumerate(lines):
        if "return await self._generate_default_analysis(instrument, timeframe)" in line and i > 650 and i < 660:
            start_idx = i + 2  # Start 2 lines after this
            break

    if start_idx:
        # Replace the problematic section with the fixed code
        fixed_section = [
            "        # Main provider initialization and data fetching block\n",
            "        try:\n",
            "            # First try to load YahooFinance if needed\n",
            "            try:\n"
        ]
        
        lines = lines[:start_idx] + fixed_section + lines[start_idx+1:]

    # Fix 2: The _generate_default_analysis indentation issue
    # Find the method definition
    method_idx = None
    for i, line in enumerate(lines):
        if "async def _generate_default_analysis" in line and i > 1380 and i < 1390:
            method_idx = i
            break

    if method_idx:
        # Ensure proper indentation (4 spaces for class methods)
        method_line = lines[method_idx].lstrip()
        lines[method_idx] = "    " + method_line

    # Write the fixed content back
    with open("trading_bot/services/chart_service/chart.py", "w") as f:
        f.writelines(lines)

    print("Manual fixes applied to trading_bot/services/chart_service/chart.py")
    print(f"Fixed try-except issue starting at line {start_idx if start_idx else 'Not found'}")
    print(f"Fixed method indentation at line {method_idx if method_idx else 'Not found'}")

if __name__ == "__main__":
    main() 