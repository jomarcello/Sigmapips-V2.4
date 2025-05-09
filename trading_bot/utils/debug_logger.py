import os
import logging
from logging.handlers import RotatingFileHandler
import json
import sys
import traceback
from datetime import datetime

class DebugLogger:
    """
    Advanced debug logger with rotating file capabilities.
    Creates persistent logs that can be analyzed by AI for debugging.
    """
    
    def __init__(self, log_name="debug", log_dir="logs", max_size_mb=10, backup_count=5):
        """
        Initialize the debug logger with rotating file handler.
        
        Args:
            log_name (str): Base name for the log file
            log_dir (str): Directory to store logs
            max_size_mb (int): Maximum size of each log file in MB
            backup_count (int): Number of backup files to keep
        """
        self.log_name = log_name
        
        # Create logs directory if it doesn't exist
        self.log_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Set up the logger
        self.logger = logging.getLogger(f"debug_{log_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create rotating file handler
        log_file = os.path.join(self.log_dir, f"{log_name}.log")
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_input(self, input_data, source="unknown"):
        """Log input data with source information"""
        try:
            if isinstance(input_data, (dict, list)):
                input_str = json.dumps(input_data, default=str)
            else:
                input_str = str(input_data)
            
            self.logger.info(f"INPUT [{source}]: {input_str}")
        except Exception as e:
            self.logger.error(f"Failed to log input: {str(e)}")
    
    def log_output(self, output_data, destination="unknown"):
        """Log output data with destination information"""
        try:
            if isinstance(output_data, (dict, list)):
                output_str = json.dumps(output_data, default=str)
            else:
                output_str = str(output_data)
            
            self.logger.info(f"OUTPUT [{destination}]: {output_str}")
        except Exception as e:
            self.logger.error(f"Failed to log output: {str(e)}")
    
    def log_error(self, error, context=None):
        """Log error with full traceback and context"""
        try:
            error_msg = f"ERROR: {str(error)}"
            if context:
                if isinstance(context, dict):
                    context_str = json.dumps(context, default=str)
                else:
                    context_str = str(context)
                error_msg += f" | Context: {context_str}"
            
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
        except Exception as e:
            self.logger.error(f"Failed to log error: {str(e)}")
    
    def log_process(self, process_name, data=None, status="running"):
        """Log process information with status"""
        try:
            msg = f"PROCESS [{process_name}] Status: {status}"
            if data:
                if isinstance(data, dict):
                    data_str = json.dumps(data, default=str)
                else:
                    data_str = str(data)
                msg += f" | Data: {data_str}"
            
            self.logger.info(msg)
        except Exception as e:
            self.logger.error(f"Failed to log process: {str(e)}")
    
    def log_variable(self, var_name, var_value):
        """Log a variable's name and value for debugging"""
        try:
            if isinstance(var_value, (dict, list)):
                value_str = json.dumps(var_value, default=str)
            else:
                value_str = str(var_value)
            
            self.logger.debug(f"VARIABLE [{var_name}]: {value_str}")
        except Exception as e:
            self.logger.error(f"Failed to log variable: {str(e)}")

# Create a default instance for easy import
default_logger = DebugLogger()

# Helper functions for quick access
def log_input(input_data, source="unknown"):
    default_logger.log_input(input_data, source)

def log_output(output_data, destination="unknown"):
    default_logger.log_output(output_data, destination)

def log_error(error, context=None):
    default_logger.log_error(error, context)

def log_process(process_name, data=None, status="running"):
    default_logger.log_process(process_name, data, status)

def log_variable(var_name, var_value):
    default_logger.log_variable(var_name, var_value) 