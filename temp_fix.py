import re

# Read the current file
with open('trading_bot/services/chart_service/chart.py', 'r') as f:
    content = f.read()

# Fix issue 1: Indentation of _generate_default_analysis
pattern1 = r'(\s+return await self\._generate_default_analysis\(instrument, timeframe\)\n)(\n)(\s\s\s\sasync def _generate_default_analysis\()'
replacement1 = r'\1\2    async def _generate_default_analysis('
content = re.sub(pattern1, replacement1, content)

# Fix issue 2: Missing except clause for try statement
pattern2 = r'(try:\s*\n\s+# Get the available data providers\s*\n\s+binance_provider = None\s*\n\s+yahoo_provider = None\s*\n\s+alltick_provider = None\s*\n)'
replacement2 = r'\1        except Exception as e:\n            logger.error(f"Error in provider setup for technical analysis: {str(e)}")\n            return await self._generate_default_analysis(instrument, timeframe)\n\n        try:\n'
content = re.sub(pattern2, replacement2, content)

# Write the fixed content back
with open('trading_bot/services/chart_service/chart.py', 'w') as f:
    f.write(content)

print("Fixed both indentation issues in chart.py.") 