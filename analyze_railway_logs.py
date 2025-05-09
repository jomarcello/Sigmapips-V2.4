#!/usr/bin/env python3
"""
Script to analyze Railway logs using our AI-powered debugging tools.
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the utilities we created
from utils.railway_log_analyzer import RailwayLogAnalyzer
from trading_bot.utils.debug_logger import DebugLogger

def main():
    """Analyze the latest deployment logs."""
    print("=== Railway Logs Analysis ===\n")
    
    # Initialize the logger
    logger = DebugLogger(log_name="railway_analysis", log_dir="logs")
    
    # Initialize the Railway log analyzer
    analyzer = RailwayLogAnalyzer(log_dir="railway_logs")
    
    # Path to the log file
    log_file = "railway_logs/latest_deployment_logs.txt"
    
    if not os.path.exists(log_file):
        print(f"Error: Log file {log_file} not found")
        return
    
    # Analyze the logs
    print(f"Analyzing logs from: {log_file}")
    analysis = analyzer.analyze_logs(log_file)
    
    if not analysis:
        print("No errors found in logs")
        return
    
    # Save analysis for AI processing
    print(f"Found {len(analysis)} potential issues")
    analysis_file = analyzer.save_analysis_for_ai(analysis)
    
    # Output results
    print(f"\nAnalysis saved to: {analysis_file}")
    print("\nPotential issues found:")
    
    for i, issue in enumerate(analysis, 1):
        print(f"\nIssue {i}:")
        print(f"  Timestamp: {issue.get('timestamp', 'Unknown')}")
        print(f"  Error: {issue.get('error_text', 'Unknown error')}")
        
        # Extract file paths and function names if available
        context = issue.get('context', '')
        file_paths = extract_file_paths(context)
        function_names = extract_function_names(context)
        
        if file_paths:
            print("  Related files:")
            for file_path in file_paths:
                print(f"    - {file_path}")
        
        if function_names:
            print("  Related functions:")
            for func_name in function_names:
                print(f"    - {func_name}")

def extract_file_paths(text):
    """Extract file paths from text"""
    import re
    
    # Look for common Python file patterns in logs
    patterns = [
        r'File "(.*?\.py)"',
        r'in (.*?\.py)',
        r'from (.*?\.py)'
    ]
    
    file_paths = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        file_paths.extend(matches)
    
    # Filter out any duplicates and non-Python files
    file_paths = [path for path in set(file_paths) if path.endswith('.py')]
    return file_paths

def extract_function_names(text):
    """Extract function names from text"""
    import re
    
    # Look for function/method names in stack traces
    patterns = [
        r'in ([a-zA-Z_][a-zA-Z0-9_]*)\(',
        r'method ([a-zA-Z_][a-zA-Z0-9_]*)'
    ]
    
    function_names = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        function_names.extend(matches)
    
    # Filter out any duplicates and common Python words
    common_words = ['the', 'in', 'is', 'and', 'not', 'for', 'with']
    function_names = [name for name in set(function_names) if name not in common_words]
    return function_names

if __name__ == "__main__":
    main() 