#!/usr/bin/env python3

with open('trading_bot/services/chart_service/chart.py.new', 'r') as f:
    content = f.read()

idx = content.find('async def _fetch_crypto_price')
if idx != -1:
    content = content[:idx] + '        except Exception as e:\n            logger.error(f"Error in default analysis: {str(e)}")\n            return f"Unable to generate analysis for {instrument}. Please try again later."\n\n    ' + content[idx:]

with open('trading_bot/services/chart_service/chart.py.new', 'w') as f:
    f.write(content) 