#!/usr/bin/env python3

with open('trading_bot/services/chart_service/chart.py.new', 'r') as f:
    lines = f.readlines()

# Find the problematic try statement
found = False
for i, line in enumerate(lines):
    if "try:" in line and i > 655 and i < 660:
        if i+1 < len(lines) and "try:" in lines[i+1]:
            # Fix the nested try statements
            lines[i] = "        # Main provider initialization block\n        try:\n"
            lines[i+1] = "            # First try to load YahooFinance\n            try:\n"
            found = True
            break

if found:
    with open('trading_bot/services/chart_service/chart.py.new', 'w') as f:
        f.writelines(lines)
    print("Fixed the nested try statements")
else:
    print("Could not find the nested try statements") 