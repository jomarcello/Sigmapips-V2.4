"""
Signal processing module for the trading bot.
"""
import os
import json
import time
from datetime import datetime

from trading_bot.utils.debug_logger import log_input, log_output, log_error, log_variable, log_process

class SignalProcessor:
    """
    Processes trading signals and executes trades.
    """
    
    def __init__(self, api_url=None):
        """Initialize the signal processor."""
        self.api_url = api_url or os.environ.get("API_URL", "https://api.example.com")
        self.signals = {}
    
    def process_signal(self, signal_id, signal_data=None):
        """
        Process a trading signal.
        
        Args:
            signal_id (str): Unique identifier for the signal
            signal_data (dict): Signal data containing trade information
            
        Returns:
            dict: Processing result
        """
        # Log input data
        log_input({"signal_id": signal_id, "signal_data": signal_data}, source="process_signal")
        log_process("signal_processing", {"signal_id": signal_id}, "started")
        
        try:
            # Validate signal data
            if not signal_data:
                error_msg = "Signal data is required"
                log_error(error_msg, {"signal_id": signal_id})
                log_process("signal_processing", {"signal_id": signal_id}, "failed")
                return {"success": False, "error": error_msg}
            
            # Log signal type
            signal_type = signal_data.get("type", "unknown")
            log_variable("signal_type", signal_type)
            
            # This is the line mentioned in the error logs
            # We add validation to prevent the TypeError
            if "price" not in signal_data or signal_data["price"] is None:
                error_msg = "Signal data is missing price information"
                log_error(error_msg, {"signal_id": signal_id})
                log_process("signal_processing", {"signal_id": signal_id}, "failed")
                return {"success": False, "error": error_msg}
            
            # Check if price is a dictionary with a value key
            if not isinstance(signal_data["price"], dict) or "value" not in signal_data["price"]:
                # Handle the case where price is a number instead of a dictionary
                if isinstance(signal_data["price"], (int, float)):
                    price_value = signal_data["price"]
                    log_variable("price_format", "direct_value")
                else:
                    error_msg = "Invalid price format in signal data"
                    log_error(error_msg, {"signal_id": signal_id, "price_data": signal_data["price"]})
                    log_process("signal_processing", {"signal_id": signal_id}, "failed")
                    return {"success": False, "error": error_msg}
            else:
                # Extract price value from the dictionary
                price_value = signal_data["price"]["value"]
                log_variable("price_format", "dictionary")
            
            # Log the price value
            log_variable("price_value", price_value)
            
            # Validate trading pair if present
            if "trading_pair" in signal_data:
                is_valid, error_msg = self.validate_trading_pair(signal_data["trading_pair"])
                if not is_valid:
                    log_error(error_msg, {"signal_id": signal_id})
                    log_process("signal_processing", {"signal_id": signal_id}, "failed")
                    return {"success": False, "error": error_msg}
            
            # Process the signal based on its type
            if signal_type == "market":
                result = self._process_market_signal(signal_id, signal_data, price_value)
            elif signal_type == "limit":
                result = self._process_limit_signal(signal_id, signal_data, price_value)
            else:
                error_msg = f"Unsupported signal type: {signal_type}"
                log_error(error_msg, {"signal_id": signal_id})
                log_process("signal_processing", {"signal_id": signal_id}, "failed")
                return {"success": False, "error": error_msg}
            
            log_output(result, "process_signal")
            log_process("signal_processing", {"signal_id": signal_id}, "completed")
            return result
            
        except Exception as e:
            log_error(e, {"signal_id": signal_id, "function": "process_signal"})
            log_process("signal_processing", {"signal_id": signal_id}, "failed")
            return {"success": False, "error": str(e)}
    
    def validate_trading_pair(self, trading_pair):
        """
        Validate a trading pair format.
        
        Args:
            trading_pair (str or dict): Trading pair string (e.g., 'BTC/USD') or dictionary
            
        Returns:
            tuple: (is_valid, error_message)
        """
        log_input({"trading_pair": trading_pair}, source="validate_trading_pair")
        log_process("trading_pair_validation", {}, "started")
        
        try:
            # Handle different trading pair formats
            if isinstance(trading_pair, dict):
                # If it's a dictionary, check for 'symbol' key
                if "symbol" not in trading_pair:
                    error_msg = "Trading pair dictionary missing 'symbol' key"
                    log_error(error_msg, {"trading_pair": trading_pair})
                    log_process("trading_pair_validation", {}, "failed")
                    return False, error_msg
                
                pair_str = trading_pair["symbol"]
            else:
                # If it's a string, use it directly
                pair_str = trading_pair
            
            # Ensure it's a string
            if not isinstance(pair_str, str):
                error_msg = f"Trading pair must be a string, got {type(pair_str).__name__}"
                log_error(error_msg, {"trading_pair": pair_str})
                log_process("trading_pair_validation", {}, "failed")
                return False, error_msg
            
            # Validate the format (should be base/quote)
            if '/' not in pair_str:
                error_msg = f"Invalid trading pair format: {pair_str}. Expected format: BASE/QUOTE"
                log_error(error_msg, {"trading_pair": pair_str})
                log_process("trading_pair_validation", {}, "failed")
                return False, error_msg
            
            # Split the pair and validate each part
            parts = pair_str.split('/')
            if len(parts) != 2:
                error_msg = f"Invalid trading pair format: {pair_str}. Expected format: BASE/QUOTE"
                log_error(error_msg, {"trading_pair": pair_str})
                log_process("trading_pair_validation", {}, "failed")
                return False, error_msg
            
            base, quote = parts
            if not base or not quote:
                error_msg = f"Invalid trading pair components: {pair_str}. Both BASE and QUOTE must be non-empty"
                log_error(error_msg, {"trading_pair": pair_str, "base": base, "quote": quote})
                log_process("trading_pair_validation", {}, "failed")
                return False, error_msg
            
            # Trading pair is valid
            log_output({"valid": True, "base": base, "quote": quote}, "validate_trading_pair")
            log_process("trading_pair_validation", {}, "completed")
            return True, None
            
        except Exception as e:
            log_error(e, {"function": "validate_trading_pair", "trading_pair": trading_pair})
            log_process("trading_pair_validation", {}, "failed")
            return False, str(e)
    
    def _process_market_signal(self, signal_id, signal_data, price_value):
        """Process a market order signal."""
        log_process("market_signal", {"signal_id": signal_id}, "started")
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Extract direction
        direction = signal_data.get("direction", "long")
        log_variable("direction", direction)
        
        # Simulate trade execution
        result = {
            "success": True,
            "signal_id": signal_id,
            "type": "market",
            "direction": direction,
            "price": price_value,
            "executed_at": datetime.now().isoformat(),
            "status": "executed"
        }
        
        log_process("market_signal", {"signal_id": signal_id}, "completed")
        return result
    
    def _process_limit_signal(self, signal_id, signal_data, price_value):
        """Process a limit order signal."""
        log_process("limit_signal", {"signal_id": signal_id}, "started")
        
        # Simulate processing time
        time.sleep(0.5)
        
        # Extract direction
        direction = signal_data.get("direction", "long")
        log_variable("direction", direction)
        
        # Simulate trade execution
        result = {
            "success": True,
            "signal_id": signal_id,
            "type": "limit",
            "direction": direction,
            "price": price_value,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }
        
        log_process("limit_signal", {"signal_id": signal_id}, "completed")
        return result
    
    def validate_signal(self, signal_data):
        """
        Validate signal data structure.
        
        Args:
            signal_data (dict): Signal data to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        log_input({"signal_data": signal_data}, source="validate_signal")
        log_process("signal_validation", {}, "started")
        
        try:
            # Check required fields
            required_fields = ["type", "direction", "price"]
            missing_fields = [field for field in required_fields if field not in signal_data]
            
            if missing_fields:
                error_msg = f"Missing required fields: {', '.join(missing_fields)}"
                log_error(error_msg, {"missing_fields": missing_fields})
                log_process("signal_validation", {}, "failed")
                return False, error_msg
            
            # Validate signal type
            valid_types = ["market", "limit", "stop"]
            if signal_data["type"] not in valid_types:
                error_msg = f"Invalid signal type: {signal_data['type']}"
                log_error(error_msg, {"valid_types": valid_types})
                log_process("signal_validation", {}, "failed")
                return False, error_msg
            
            # Validate direction
            valid_directions = ["long", "short"]
            if signal_data["direction"] not in valid_directions:
                error_msg = f"Invalid direction: {signal_data['direction']}"
                log_error(error_msg, {"valid_directions": valid_directions})
                log_process("signal_validation", {}, "failed")
                return False, error_msg
            
            # Validate price
            if isinstance(signal_data["price"], dict):
                if "value" not in signal_data["price"]:
                    error_msg = "Price dictionary missing 'value' key"
                    log_error(error_msg, {"price": signal_data["price"]})
                    log_process("signal_validation", {}, "failed")
                    return False, error_msg
                
                price_value = signal_data["price"]["value"]
            else:
                price_value = signal_data["price"]
            
            if not isinstance(price_value, (int, float)) or price_value <= 0:
                error_msg = f"Invalid price value: {price_value}"
                log_error(error_msg, {"price_value": price_value})
                log_process("signal_validation", {}, "failed")
                return False, error_msg
            
            # Validate trading pair if present
            if "trading_pair" in signal_data:
                is_valid, error_msg = self.validate_trading_pair(signal_data["trading_pair"])
                if not is_valid:
                    log_error(error_msg, {"signal_data": signal_data})
                    log_process("signal_validation", {}, "failed")
                    return False, error_msg
            
            # Signal is valid
            log_output({"valid": True}, "validate_signal")
            log_process("signal_validation", {}, "completed")
            return True, None
            
        except Exception as e:
            log_error(e, {"function": "validate_signal"})
            log_process("signal_validation", {}, "failed")
            return False, str(e)

# Create a default instance
signal_processor = SignalProcessor() 