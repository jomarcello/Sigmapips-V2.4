#!/usr/bin/env python3
"""
Railway AI Debugging Tool

This script combines Railway log analysis with AI-powered debugging.
It fetches logs from Railway, analyzes them for errors, and adds targeted debugging
to help identify and fix issues.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
import traceback
from typing import List, Dict, Any, Optional

# Import the utilities we created
from utils.railway_log_analyzer import RailwayLogAnalyzer
from utils.debug_integrator import DebugIntegrator
from trading_bot.utils.debug_logger import DebugLogger, log_input, log_output, log_error, log_process, log_variable

class RailwayAIDebugger:
    """
    A tool for AI-powered debugging of Railway deployments.
    """
    
    def __init__(self, service_name: str = None, project_name: str = None, log_dir: str = "railway_ai_debug"):
        """
        Initialize the Railway AI debugger.
        
        Args:
            service_name (str): Name of the Railway service to debug
            project_name (str): Name of the Railway project
            log_dir (str): Directory to store debug logs
        """
        # Initialize the logger
        self.logger = DebugLogger(log_name="railway_ai_debug", log_dir=log_dir)
        
        # Initialize the Railway log analyzer
        self.analyzer = RailwayLogAnalyzer(
            service_name=service_name,
            project_name=project_name,
            log_dir=log_dir
        )
        
        # Initialize the debug integrator
        self.integrator = DebugIntegrator(log_name="railway_integrator")
        
        # Store configuration
        self.service_name = service_name
        self.project_name = project_name
        self.log_dir = log_dir
    
    def debug_issue(self, issue_description: str, debug_duration: int = 1):
        """
        Debug a specific issue by fetching logs and applying targeted debugging.
        
        Args:
            issue_description (str): Description of the issue to debug
            debug_duration (int): Duration in hours of logs to analyze
            
        Returns:
            Dict: Debug results with analysis and recommendations
        """
        log_process(f"debug_issue", {"issue": issue_description, "duration": debug_duration}, "started")
        
        try:
            # Step 1: Fetch recent logs from Railway
            self.logger.logger.info(f"Step 1: Fetching Railway logs for the past {debug_duration} hour(s)")
            log_file = self.analyzer.fetch_logs(hours=debug_duration)
            
            if not log_file or not os.path.exists(log_file):
                error_msg = "Failed to fetch Railway logs"
                self.logger.logger.error(error_msg)
                log_error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Step 2: Analyze logs for errors
            self.logger.logger.info("Step 2: Analyzing logs for errors")
            analysis = self.analyzer.analyze_logs(log_file)
            
            if not analysis:
                self.logger.logger.info("No errors found in logs")
                return {"success": True, "errors_found": False, "message": "No errors found in logs"}
            
            # Step 3: Save analysis for AI processing
            self.logger.logger.info(f"Step 3: Found {len(analysis)} potential issues, saving for AI analysis")
            analysis_file = self.analyzer.save_analysis_for_ai(analysis)
            
            # Step 4: Prepare debug recommendations
            self.logger.logger.info("Step 4: Preparing debug recommendations")
            recommendations = self._generate_debug_recommendations(analysis, issue_description)
            
            # Step 5: Output results
            results = {
                "success": True,
                "errors_found": True,
                "num_issues": len(analysis),
                "log_file": log_file,
                "analysis_file": analysis_file,
                "recommendations": recommendations
            }
            
            log_output(results, "debug_issue")
            log_process(f"debug_issue", {"issue": issue_description}, "completed")
            
            return results
            
        except Exception as e:
            error_msg = f"Error debugging issue: {str(e)}"
            self.logger.logger.error(error_msg)
            self.logger.logger.error(traceback.format_exc())
            log_error(e, {"issue": issue_description})
            
            return {"success": False, "error": error_msg}
    
    def start_live_debugging(self, issue_description: str):
        """
        Start live debugging session with continuous log monitoring.
        
        Args:
            issue_description (str): Description of the issue to debug
            
        Returns:
            Dict: Information about the live debugging session
        """
        log_process(f"live_debugging", {"issue": issue_description}, "started")
        
        try:
            # Step 1: Start streaming logs
            self.logger.logger.info("Starting Railway log streaming")
            thread, log_file = self.analyzer.start_log_streaming()
            
            if not thread or not log_file:
                error_msg = "Failed to start Railway log streaming"
                self.logger.logger.error(error_msg)
                log_error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Step 2: Return session info
            session_info = {
                "success": True,
                "session_started": True,
                "log_file": log_file,
                "issue": issue_description,
                "timestamp": datetime.now().isoformat(),
                "message": "Live debugging session started. Logs are being collected. Run analyze_live_session when the issue occurs."
            }
            
            log_output(session_info, "start_live_debugging")
            return session_info
            
        except Exception as e:
            error_msg = f"Error starting live debugging: {str(e)}"
            self.logger.logger.error(error_msg)
            self.logger.logger.error(traceback.format_exc())
            log_error(e, {"issue": issue_description})
            
            return {"success": False, "error": error_msg}
    
    def analyze_live_session(self, log_file: str, issue_description: str):
        """
        Analyze logs from a live debugging session.
        
        Args:
            log_file (str): Path to the log file
            issue_description (str): Description of the issue
            
        Returns:
            Dict: Analysis results
        """
        log_process(f"analyze_live_session", {"log_file": log_file, "issue": issue_description}, "started")
        
        try:
            # Step 1: Analyze the log file
            self.logger.logger.info(f"Analyzing log file: {log_file}")
            analysis = self.analyzer.analyze_logs(log_file)
            
            if not analysis:
                self.logger.logger.info("No errors found in logs")
                return {"success": True, "errors_found": False, "message": "No errors found in logs"}
            
            # Step 2: Save analysis for AI processing
            self.logger.logger.info(f"Found {len(analysis)} potential issues, saving for AI analysis")
            analysis_file = self.analyzer.save_analysis_for_ai(analysis)
            
            # Step 3: Prepare debug recommendations
            self.logger.logger.info("Preparing debug recommendations")
            recommendations = self._generate_debug_recommendations(analysis, issue_description)
            
            # Step 4: Output results
            results = {
                "success": True,
                "errors_found": True,
                "num_issues": len(analysis),
                "log_file": log_file,
                "analysis_file": analysis_file,
                "recommendations": recommendations
            }
            
            log_output(results, "analyze_live_session")
            log_process(f"analyze_live_session", {}, "completed")
            
            return results
            
        except Exception as e:
            error_msg = f"Error analyzing live session: {str(e)}"
            self.logger.logger.error(error_msg)
            self.logger.logger.error(traceback.format_exc())
            log_error(e, {"log_file": log_file, "issue": issue_description})
            
            return {"success": False, "error": error_msg}
    
    def add_targeted_logging(self, file_path: str, function_name: str = None):
        """
        Add targeted logging to a specific file or function.
        
        Args:
            file_path (str): Path to the file to add logging to
            function_name (str): Specific function to add logging to
            
        Returns:
            Dict: Results of adding logging
        """
        log_process(f"add_targeted_logging", {"file": file_path, "function": function_name}, "started")
        
        try:
            # Step 1: Analyze the file
            self.logger.logger.info(f"Analyzing file: {file_path}")
            analysis = self.integrator.analyze_file(file_path)
            
            if not analysis:
                error_msg = f"Failed to analyze file: {file_path}"
                self.logger.logger.error(error_msg)
                log_error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Step 2: If function_name is specified, only add logging to that function
            if function_name:
                # Find the function in the analysis
                target_function = None
                for func in analysis.get('functions', []):
                    if func['name'] == function_name:
                        target_function = func
                        break
                
                # If not found in functions, check class methods
                if not target_function:
                    for cls in analysis.get('classes', []):
                        for method in cls.get('methods', []):
                            if method['name'] == function_name:
                                target_function = method
                                target_function['class'] = cls['name']
                                break
                        if target_function:
                            break
                
                if not target_function:
                    error_msg = f"Function {function_name} not found in {file_path}"
                    self.logger.logger.error(error_msg)
                    log_error(error_msg)
                    return {"success": False, "error": error_msg}
                
                # Create a filtered analysis with only the target function
                filtered_analysis = {
                    'file_path': analysis['file_path'],
                    'imports': analysis['imports'],
                    'has_debug_logger': analysis['has_debug_logger'],
                    'functions': [],
                    'classes': []
                }
                
                if 'class' in target_function:
                    # It's a method
                    class_name = target_function['class']
                    for cls in analysis.get('classes', []):
                        if cls['name'] == class_name:
                            filtered_cls = {
                                'name': cls['name'],
                                'line': cls['line'],
                                'methods': []
                            }
                            for method in cls.get('methods', []):
                                if method['name'] == function_name:
                                    filtered_cls['methods'].append(method)
                            filtered_analysis['classes'].append(filtered_cls)
                else:
                    # It's a function
                    filtered_analysis['functions'].append(target_function)
                
                # Apply logging using the filtered analysis
                inserts = self.integrator.generate_logging_code(filtered_analysis)
            else:
                # Apply logging to the entire file
                inserts = self.integrator.generate_logging_code(analysis)
            
            # Step 3: Apply the logging code
            self.logger.logger.info(f"Applying logging code to {file_path}")
            modified_code = self.integrator.apply_logging(file_path, dry_run=False)
            
            if not modified_code:
                self.logger.logger.warning(f"No changes made to {file_path}")
                return {"success": True, "changes": False, "message": "No changes were necessary"}
            
            # Step 4: Output results
            results = {
                "success": True,
                "changes": True,
                "file_path": file_path,
                "function_name": function_name,
                "num_inserts": len(inserts) if inserts else 0,
                "message": f"Added logging to {'function ' + function_name if function_name else 'file'}"
            }
            
            log_output(results, "add_targeted_logging")
            log_process(f"add_targeted_logging", {}, "completed")
            
            return results
            
        except Exception as e:
            error_msg = f"Error adding targeted logging: {str(e)}"
            self.logger.logger.error(error_msg)
            self.logger.logger.error(traceback.format_exc())
            log_error(e, {"file": file_path, "function": function_name})
            
            return {"success": False, "error": error_msg}
    
    def _generate_debug_recommendations(self, analysis: List[Dict[str, Any]], issue_description: str) -> Dict[str, Any]:
        """
        Generate debug recommendations based on log analysis.
        
        Args:
            analysis (List[Dict]): Log analysis results
            issue_description (str): Description of the issue
            
        Returns:
            Dict: Debug recommendations
        """
        # This function would ideally use AI to analyze the logs and generate recommendations
        # For now, we'll implement a simple heuristic-based approach
        recommendations = {
            "possible_issues": [],
            "suggested_files": [],
            "suggested_functions": []
        }
        
        # Process each error found
        for issue in analysis:
            error_text = issue.get('error_text', '')
            context = issue.get('context', '')
            
            # Extract file paths and function names from error context
            file_paths = self._extract_file_paths(context)
            function_names = self._extract_function_names(context)
            
            # Add to suggestions if not already present
            for file_path in file_paths:
                if file_path not in recommendations["suggested_files"]:
                    recommendations["suggested_files"].append(file_path)
            
            for function_name in function_names:
                if function_name not in recommendations["suggested_functions"]:
                    recommendations["suggested_functions"].append(function_name)
            
            # Add the issue to possible issues
            recommendations["possible_issues"].append({
                "error_text": error_text,
                "related_files": file_paths,
                "related_functions": function_names,
                "timestamp": issue.get('timestamp')
            })
        
        return recommendations
    
    def _extract_file_paths(self, text: str) -> List[str]:
        """Extract file paths from text"""
        import re
        
        # Look for common Python file patterns in logs
        # This is a simple implementation and would need to be improved
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
    
    def _extract_function_names(self, text: str) -> List[str]:
        """Extract function names from text"""
        import re
        
        # Look for function/method names in stack traces
        # This is a simple implementation and would need to be improved
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

def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(description="Railway AI Debugging Tool")
    
    parser.add_argument("--service", "-s", help="Railway service name")
    parser.add_argument("--project", "-p", help="Railway project name")
    parser.add_argument("--log-dir", "-d", default="railway_ai_debug", help="Directory to store debug logs")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Debug issue command
    debug_parser = subparsers.add_parser("debug", help="Debug a specific issue")
    debug_parser.add_argument("issue", help="Description of the issue to debug")
    debug_parser.add_argument("--duration", "-t", type=int, default=1, help="Duration in hours of logs to analyze")
    
    # Live debugging command
    live_parser = subparsers.add_parser("live", help="Start a live debugging session")
    live_parser.add_argument("issue", help="Description of the issue to debug")
    
    # Analyze live session command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a live debugging session")
    analyze_parser.add_argument("log_file", help="Path to the log file")
    analyze_parser.add_argument("issue", help="Description of the issue")
    
    # Add targeted logging command
    log_parser = subparsers.add_parser("log", help="Add targeted logging to a file or function")
    log_parser.add_argument("file", help="Path to the file to add logging to")
    log_parser.add_argument("--function", "-f", help="Specific function to add logging to")
    
    args = parser.parse_args()
    
    # Create the debugger
    debugger = RailwayAIDebugger(
        service_name=args.service,
        project_name=args.project,
        log_dir=args.log_dir
    )
    
    # Run the appropriate command
    if args.command == "debug":
        results = debugger.debug_issue(args.issue, args.duration)
        print(json.dumps(results, indent=2))
    
    elif args.command == "live":
        results = debugger.start_live_debugging(args.issue)
        print(json.dumps(results, indent=2))
    
    elif args.command == "analyze":
        results = debugger.analyze_live_session(args.log_file, args.issue)
        print(json.dumps(results, indent=2))
    
    elif args.command == "log":
        results = debugger.add_targeted_logging(args.file, args.function)
        print(json.dumps(results, indent=2))
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 