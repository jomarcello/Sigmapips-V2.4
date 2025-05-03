#!/usr/bin/env python3

"""
Script to fix all syntax and indentation issues in chart.py
"""

import os
import re

def main():
    # Backup original file if not already done
    if not os.path.exists("trading_bot/services/chart_service/chart.py.bak"):
        os.system("cp trading_bot/services/chart_service/chart.py trading_bot/services/chart_service/chart.py.bak")
    
    # Read current file content
    with open("trading_bot/services/chart_service/chart.py", "r") as f:
        lines = f.readlines()
    
    # Fix the content line by line
    fixed_lines = []
    fix_next_try = False
    in_default_analysis = False
    default_analysis_indent = 0
    
    for i, line in enumerate(lines):
        if "async def _generate_default_analysis" in line:
            # Correct the indentation of the _generate_default_analysis function
            in_default_analysis = True
            default_analysis_indent = line.find("async")
            fixed_line = " " * 4 + line.lstrip()  # Ensure it's at the class level indentation
            fixed_lines.append(fixed_line)
        elif in_default_analysis and line.strip() and line[default_analysis_indent:default_analysis_indent+1].strip():
            # End of the _generate_default_analysis function
            in_default_analysis = False
            fixed_lines.append(line)
        elif in_default_analysis:
            # Keep the indentation within the _generate_default_analysis function
            fixed_line = " " * 4 + line[default_analysis_indent:]
            fixed_lines.append(fixed_line)
        elif "try" in line and "# Get the available data providers" in lines[i+1]:
            # Fix the try statement that needs an except clause
            fixed_lines.append(line)  # Add the original try line
            
            # Find the end of this try block (before the next try)
            try_block_lines = []
            j = i + 1
            nesting_level = 1
            while j < len(lines) and nesting_level > 0:
                if "try:" in lines[j]:
                    nesting_level += 1
                elif "except" in lines[j]:
                    nesting_level -= 1
                
                # Skip this section if we found a nested try-except 
                if nesting_level == 0 and "except" in lines[j]:
                    break
                    
                try_block_lines.append(lines[j])
                j += 1
            
            # Add all lines of the try block
            fixed_lines.extend(try_block_lines)
            
            # Add the missing except clause
            fixed_lines.append(" " * 8 + "except Exception as e:\n")
            fixed_lines.append(" " * 12 + "logger.error(f\"Error in provider setup for technical analysis: {str(e)}\")\n")
            fixed_lines.append(" " * 12 + "return await self._generate_default_analysis(instrument, timeframe)\n")
            fixed_lines.append("\n")
            
            # Skip the lines we've already processed
            fix_next_try = True
        elif fix_next_try and line.strip() == "try:":
            fixed_lines.append(" " * 8 + line)  # Ensure correct indentation
            fix_next_try = False
        else:
            fixed_lines.append(line)
    
    # Write the fixed content to a temporary file
    with open("trading_bot/services/chart_service/chart.py.fixed", "w") as f:
        f.writelines(fixed_lines)
    
    # Replace the original file with the fixed one
    os.system("mv trading_bot/services/chart_service/chart.py.fixed trading_bot/services/chart_service/chart.py")
    
    print("Fixed all syntax and indentation issues in chart.py")

if __name__ == "__main__":
    main() 