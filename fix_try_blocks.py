#!/usr/bin/env python3

"""
Script to fix all incomplete try blocks in chart.py
"""

def main():
    with open('trading_bot/services/chart_service/chart.py', 'r') as f:
        content = f.read()
    
    # Find the incomplete try block by looking for `try:` that's not followed by an `except` or `finally`
    # and is followed by the next method declaration
    pattern = 'try:\n            # Get the current time for cache lookups'
    replacement = """try:
            # Get the current time for cache lookups
        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            return await self._generate_default_analysis(instrument, timeframe)
            
        try:"""
    
    fixed_content = content.replace(pattern, replacement)
    
    # Fix the second issue with _generate_default_analysis
    pattern2 = 'return await self._generate_default_analysis(instrument, timeframe)\n\n    async def _generate_default_analysis'
    replacement2 = 'return await self._generate_default_analysis(instrument, timeframe)\n\nasync def _generate_default_analysis'
    
    fixed_content = fixed_content.replace(pattern2, replacement2)
    
    with open('trading_bot/services/chart_service/chart.py', 'w') as f:
        f.write(fixed_content)
    
    print('Fixed incomplete try blocks in chart.py')

if __name__ == "__main__":
    main() 