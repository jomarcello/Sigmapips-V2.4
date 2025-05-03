#!/usr/bin/env python3
import re

# Read the chart.py file
file_path = "trading_bot/services/chart_service/chart.py"
with open(file_path, "r") as f:
    content = f.read()

# Fix the try statement at line 657 that doesn't have an except/finally clause
try_without_except_pattern = r"(\s+try:\s+)(\s+try:)"
try_with_except_replacement = r"\1    except Exception as e:\n        logger.error(f\"Outer provider initialization error: {str(e)}\")\n        return await self._generate_default_analysis(instrument, timeframe)\n\2"

content = re.sub(try_without_except_pattern, try_with_except_replacement, content)

# Fix the indentation for _generate_default_analysis method
# First check if there's an indentation issue
indentation_pattern = r"(\s+return await self\._generate_default_analysis\(instrument, timeframe\)\s*\n+)(\s*)async def _generate_default_analysis\("
matches = re.findall(indentation_pattern, content)

if matches:
    print("Found indentation issue with _generate_default_analysis")
    # Ensure there's a proper class-level indentation (4 spaces)
    fixed_content = re.sub(indentation_pattern, r"\1    async def _generate_default_analysis(", content)
    
    # Write the fixed content back to the file
    with open(file_path, "w") as f:
        f.write(fixed_content)
    print(f"Fixed indentation issue in {file_path}")
else:
    print("No indentation issue found with _generate_default_analysis")
    # Write back the content with just the try-except fix
    with open(file_path, "w") as f:
        f.write(content)
    print(f"Fixed only the try statement in {file_path}") 