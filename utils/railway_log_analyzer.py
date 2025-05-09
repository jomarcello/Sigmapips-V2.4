import os
import subprocess
import json
import time
import re
from datetime import datetime, timedelta
import sys
import tempfile
import threading

# Add parent directory to path to import debug_logger
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trading_bot.utils.debug_logger import DebugLogger

class RailwayLogAnalyzer:
    """
    A utility class to fetch logs from Railway and prepare them for AI analysis.
    Integrates with the DebugLogger for local logging.
    """
    
    def __init__(self, service_name=None, project_name=None, environment=None, log_dir="railway_logs"):
        """
        Initialize the Railway log analyzer.
        
        Args:
            service_name (str): Name of the Railway service to fetch logs from
            project_name (str): Name of the Railway project
            environment (str): Railway environment (e.g., 'production')
            log_dir (str): Directory to store fetched logs
        """
        self.service_name = service_name
        self.project_name = project_name
        self.environment = environment
        
        # Set up logger
        self.logger = DebugLogger(log_name="railway_analyzer", log_dir=log_dir)
        self.log_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Check if Railway CLI is installed
        self._check_railway_cli()
    
    def _check_railway_cli(self):
        """Check if Railway CLI is installed and working"""
        try:
            result = subprocess.run(
                ["railway", "version"], 
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                self.logger.logger.warning("Railway CLI not installed or not working properly.")
                self.logger.logger.warning(f"Error: {result.stderr}")
            else:
                self.logger.logger.info(f"Railway CLI detected: {result.stdout.strip()}")
        except Exception as e:
            self.logger.logger.error(f"Error checking Railway CLI: {str(e)}")
    
    def fetch_logs(self, hours=1, output_file=None, service=None):
        """
        Fetch logs from Railway for the specified time period.
        
        Args:
            hours (int): Number of hours of logs to fetch
            output_file (str): File to save logs to (default: timestamped file)
            service (str): Override the default service name
            
        Returns:
            str: Path to the log file
        """
        service_name = service or self.service_name
        
        if not service_name:
            self.logger.logger.error("No service name provided.")
            return None
        
        # Create timestamped filename if none provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.log_dir, f"railway_logs_{timestamp}.txt")
        
        # Build Railway CLI command
        cmd = ["railway", "logs"]
        
        if self.project_name:
            cmd.extend(["--project", self.project_name])
        
        if self.environment:
            cmd.extend(["--environment", self.environment])
        
        if service_name:
            cmd.extend(["--service", service_name])
        
        # Add time filter
        if hours:
            cmd.extend(["--since", f"{hours}h"])
        
        self.logger.logger.info(f"Fetching Railway logs with command: {' '.join(cmd)}")
        
        try:
            # Run the command and capture output
            with open(output_file, 'w') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Wait for a short time to see if there's an immediate error
                time.sleep(2)
                
                # Check if process is still running
                if process.poll() is not None:
                    stderr = process.stderr.read()
                    if stderr:
                        self.logger.logger.error(f"Error fetching logs: {stderr}")
                        return None
                
                # Process is running, will write to file
                self.logger.logger.info(f"Railway logs being written to {output_file}")
                
                # Let it run for a bit to collect logs
                time.sleep(10)
                
                # Terminate the process
                process.terminate()
                
                return output_file
                
        except Exception as e:
            self.logger.logger.error(f"Failed to fetch Railway logs: {str(e)}")
            return None
    
    def start_log_streaming(self, service=None, output_file=None):
        """
        Start streaming logs from Railway in a background thread.
        
        Args:
            service (str): Override the default service name
            output_file (str): File to save logs to (default: timestamped file)
            
        Returns:
            tuple: (thread, output_file) - The background thread and log file path
        """
        service_name = service or self.service_name
        
        if not service_name:
            self.logger.logger.error("No service name provided.")
            return None, None
        
        # Create timestamped filename if none provided
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.log_dir, f"railway_stream_{timestamp}.txt")
        
        # Build Railway CLI command
        cmd = ["railway", "logs"]
        
        if self.project_name:
            cmd.extend(["--project", self.project_name])
        
        if self.environment:
            cmd.extend(["--environment", self.environment])
        
        if service_name:
            cmd.extend(["--service", service_name])
        
        self.logger.logger.info(f"Starting Railway log streaming with command: {' '.join(cmd)}")
        
        # Create a file for logging
        log_file = open(output_file, 'w')
        
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Create a thread to monitor the process
        def monitor_process():
            try:
                # Check if there's an immediate error
                time.sleep(2)
                if process.poll() is not None:
                    stderr = process.stderr.read()
                    if stderr:
                        self.logger.logger.error(f"Error streaming logs: {stderr}")
                    return
                
                self.logger.logger.info(f"Railway log streaming active to {output_file}")
                
                # Wait for the process to complete
                process.wait()
            except Exception as e:
                self.logger.logger.error(f"Error in log streaming thread: {str(e)}")
            finally:
                log_file.close()
        
        # Start the monitoring thread
        thread = threading.Thread(target=monitor_process)
        thread.daemon = True
        thread.start()
        
        return thread, output_file
    
    def stop_log_streaming(self, thread):
        """
        Stop a log streaming thread.
        
        Args:
            thread: The thread returned from start_log_streaming
        """
        if thread and thread.is_alive():
            # The thread will end when the process is terminated
            # This is handled by the OS when the Python process exits
            self.logger.logger.info("Log streaming will be stopped when the process exits")
    
    def analyze_logs(self, log_file, error_pattern=None, context_lines=10):
        """
        Analyze logs to extract errors and their context.
        
        Args:
            log_file (str): Path to the log file
            error_pattern (str): Regex pattern to identify errors
            context_lines (int): Number of lines to include before and after each error
            
        Returns:
            list: List of dictionaries containing errors and their context
        """
        if not os.path.exists(log_file):
            self.logger.logger.error(f"Log file not found: {log_file}")
            return []
        
        # Default error pattern if none provided
        if not error_pattern:
            error_pattern = r'(error|exception|fail|traceback|critical|fatal|crash)'
        
        error_regex = re.compile(error_pattern, re.IGNORECASE)
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            results = []
            for i, line in enumerate(lines):
                if error_regex.search(line):
                    # Get context lines
                    start = max(0, i - context_lines)
                    end = min(len(lines), i + context_lines + 1)
                    
                    context = {
                        'error_line': i + 1,
                        'error_text': line.strip(),
                        'context': ''.join(lines[start:end]),
                        'timestamp': self._extract_timestamp(line)
                    }
                    
                    results.append(context)
            
            self.logger.logger.info(f"Found {len(results)} potential errors in log file")
            return results
        
        except Exception as e:
            self.logger.logger.error(f"Error analyzing logs: {str(e)}")
            return []
    
    def _extract_timestamp(self, log_line):
        """Extract timestamp from a log line if present"""
        # Common timestamp patterns
        patterns = [
            r'(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)',
            r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, log_line)
            if match:
                return match.group(1)
        
        return None
    
    def save_analysis_for_ai(self, analysis_results, output_file=None):
        """
        Save analysis results in a format suitable for AI processing.
        
        Args:
            analysis_results (list): Results from analyze_logs
            output_file (str): File to save results to
            
        Returns:
            str: Path to the output file
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.log_dir, f"ai_analysis_{timestamp}.json")
        
        try:
            with open(output_file, 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            self.logger.logger.info(f"Analysis saved to {output_file}")
            return output_file
        except Exception as e:
            self.logger.logger.error(f"Error saving analysis: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    analyzer = RailwayLogAnalyzer(service_name="sigmapips-bot")
    
    # Fetch logs from the last hour
    log_file = analyzer.fetch_logs(hours=1)
    
    if log_file:
        # Analyze logs
        analysis = analyzer.analyze_logs(log_file)
        
        # Save analysis for AI
        analyzer.save_analysis_for_ai(analysis)
        
        print(f"Log analysis complete. Found {len(analysis)} potential issues.") 