#!/usr/bin/env python3

# Read the file into lines
with open("trading_bot/services/chart_service/chart.py", "r") as f:
    lines = f.readlines()

# Fix 1: The try-without-except issue
# Find the line number of the problematic try
try_line = None
for i, line in enumerate(lines):
    if "try:" in line and i > 655 and i < 665:
        if "try:" in lines[i+1]:  # If the next line also has 'try:'
            try_line = i
            break

if try_line is not None:
    print(f"Found problematic try at line {try_line+1}")
    
    # Remove the outer try and replace with a proper try-except structure
    outer_indentation = lines[try_line].split("try:")[0]
    lines[try_line] = f"{outer_indentation}# Main provider initialization\n"
    lines.insert(try_line+1, f"{outer_indentation}try:\n")

# Fix 2: The _generate_default_analysis indentation issue
# Look for the method definition
method_line = None
for i, line in enumerate(lines):
    if "_generate_default_analysis" in line and "async def" in line:
        method_line = i
        break

if method_line is not None:
    print(f"Found _generate_default_analysis at line {method_line+1}")
    
    # Check and fix indentation if needed
    if not lines[method_line].startswith("    "):
        # Fix indentation - ensure it's a class method (4 spaces)
        lines[method_line] = "    " + lines[method_line].lstrip()
    
    # Also check if the method has a matching except-finally for its try block
    method_content = "".join(lines[method_line:])
    if method_content.count("try:") > method_content.count("except"):
        print("Method missing matching except clause - adding it")
        
        # Find the end of the method (next method definition or end of file)
        end_line = len(lines)
        for i in range(method_line + 1, len(lines)):
            if "async def" in lines[i] or "def " in lines[i]:
                end_line = i
                break
        
        # Add the missing except clause before the end of the method
        lines.insert(end_line, "        except Exception as e:\n")
        lines.insert(end_line + 1, "            logger.error(f\"Error in default analysis: {str(e)}\")\n")
        lines.insert(end_line + 2, "            return f\"Unable to generate analysis for {instrument}. Please try again later.\"\n")
        lines.insert(end_line + 3, "\n")

# Write the modified content back
with open("trading_bot/services/chart_service/chart.py", "w") as f:
    f.writelines(lines)

print("Manually patched trading_bot/services/chart_service/chart.py") 