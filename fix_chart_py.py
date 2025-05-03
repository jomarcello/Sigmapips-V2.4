#!/usr/bin/env python3

# This script fixes the indentation issue and the try without except/finally clause in chart.py
file_path = "trading_bot/services/chart_service/chart.py"

# Read the content
with open(file_path, "r") as f:
    content = f.readlines()

# Find the problematic area - the try without except error
in_outer_try = False
error_line = None
for i, line in enumerate(content):
    if "try:" in line and i > 650 and i < 660:
        if in_outer_try:
            # Found a nested try without except/finally for the first try
            error_line = i
            break
        in_outer_try = True

# Add an except/finally to the outer try before the inner try starts
if error_line:
    # Add an except clause just before the inner try
    if "try:" in content[error_line]:
        # Convert the inner try line to a string with proper indentation
        inner_try_line = content[error_line]
        indentation = content[error_line].split("try:")[0]
        
        # Create a fixed version with a proper except clause before the inner try
        fixed_lines = []
        fixed_lines.append(f"{indentation}except Exception as e:\n")
        fixed_lines.append(f"{indentation}    logger.error(f\"Outer provider initialization error: {{str(e)}}\")\n")
        fixed_lines.append(f"{indentation}    return await self._generate_default_analysis(instrument, timeframe)\n")
        fixed_lines.append("\n")
        fixed_lines.append(f"{indentation}# Main provider initialization block\n")
        fixed_lines.append(f"{indentation}try:\n")
        
        # Update the content
        content = content[:error_line] + fixed_lines + content[error_line+1:]

# Find the indentation issue with _generate_default_analysis
# Look for the pattern of return _generate_default_analysis followed by the method definition
method_def_line = None
for i, line in enumerate(content):
    if "_generate_default_analysis" in line and "async def" in line and i > 1000 and i < 1500:
        method_def_line = i
        break

# Fix the indentation if we found the method definition
if method_def_line:
    current_indent = len(content[method_def_line]) - len(content[method_def_line].lstrip())
    if current_indent != 4:  # If not properly indented at class level
        # Fix indentation to 4 spaces (class method level)
        method_line = content[method_def_line].lstrip()
        content[method_def_line] = "    " + method_line  # 4 spaces for class method

# Write the fixed content back
with open(file_path, "w") as f:
    f.writelines(content)

print(f"Fixed indentation and try-except issues in {file_path}")
print(f"Try without except error fixed at line ~{error_line if error_line else 'Not found'}")
print(f"Method indentation fixed at line ~{method_def_line if method_def_line else 'Not found'}") 