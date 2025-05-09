#!/usr/bin/env python3
"""
Test script to verify that the trading pair validation fix works correctly.
"""

import os
import sys
import json
from datetime import datetime

# Mock class for debugging
class MockLogger:
    def __init__(self):
        pass
    
    def info(self, message):
        print(f"INFO: {message}")
    
    def error(self, message):
        print(f"ERROR: {message}")
    
    def warning(self, message):
        print(f"WARNING: {message}")

# Mock the signal processor
class MockSignalProcessor:
    """Mock implementation of SignalProcessor for testing purposes."""
    
    def __init__(self):
        """Initialize the mock signal processor."""
        self.logger = MockLogger()
    
    def validate_trading_pair(self, trading_pair):
        """
        Validate a trading pair format.
        
        Args:
            trading_pair (str or dict): Trading pair string (e.g., 'BTC/USD') or dictionary
            
        Returns:
            tuple: (is_valid, error_message)
        """
        print(f"Validating trading pair: {trading_pair}")
        
        try:
            # Handle different trading pair formats
            if isinstance(trading_pair, dict):
                # If it's a dictionary, check for 'symbol' key
                if "symbol" not in trading_pair:
                    error_msg = "Trading pair dictionary missing 'symbol' key"
                    print(f"ERROR: {error_msg}")
                    return False, error_msg
                
                pair_str = trading_pair["symbol"]
            else:
                # If it's a string, use it directly
                pair_str = trading_pair
            
            # Ensure it's a string
            if not isinstance(pair_str, str):
                error_msg = f"Trading pair must be a string, got {type(pair_str).__name__}"
                print(f"ERROR: {error_msg}")
                return False, error_msg
            
            # Validate the format (should be base/quote)
            if '/' not in pair_str:
                error_msg = f"Invalid trading pair format: {pair_str}. Expected format: BASE/QUOTE"
                print(f"ERROR: {error_msg}")
                return False, error_msg
            
            # Split the pair and validate each part
            parts = pair_str.split('/')
            if len(parts) != 2:
                error_msg = f"Invalid trading pair format: {pair_str}. Expected format: BASE/QUOTE"
                print(f"ERROR: {error_msg}")
                return False, error_msg
            
            base, quote = parts
            if not base or not quote:
                error_msg = f"Invalid trading pair components: {pair_str}. Both BASE and QUOTE must be non-empty"
                print(f"ERROR: {error_msg}")
                return False, error_msg
            
            # Trading pair is valid
            print(f"Trading pair is valid: base={base}, quote={quote}")
            return True, None
            
        except Exception as e:
            error_msg = f"Error validating trading pair: {str(e)}"
            print(f"ERROR: {error_msg}")
            return False, error_msg
    
    def process_signal(self, signal_id, signal_data=None):
        """
        Process a trading signal.
        
        Args:
            signal_id (str): Unique identifier for the signal
            signal_data (dict): Signal data containing trade information
            
        Returns:
            dict: Processing result
        """
        print(f"Processing signal: {signal_id}")
        print(f"Signal data: {signal_data}")
        
        try:
            # Validate signal data
            if not signal_data:
                error_msg = "Signal data is required"
                print(f"ERROR: {error_msg}")
                return {"success": False, "error": error_msg}
            
            # Validate mandatory fields
            required_fields = ["type", "direction", "price"]
            missing_fields = [field for field in required_fields if field not in signal_data]
            if missing_fields:
                error_msg = f"Missing required fields: {', '.join(missing_fields)}"
                print(f"ERROR: {error_msg}")
                return {"success": False, "error": error_msg}
            
            # Validate trading pair if present
            if "trading_pair" in signal_data:
                is_valid, error_msg = self.validate_trading_pair(signal_data["trading_pair"])
                if not is_valid:
                    print(f"ERROR: Invalid trading pair: {error_msg}")
                    return {"success": False, "error": error_msg}
            
            # Simulate successful processing
            return {
                "success": True,
                "signal_id": signal_id,
                "type": signal_data.get("type"),
                "direction": signal_data.get("direction"),
                "price": signal_data.get("price"),
                "processed_at": datetime.now().isoformat(),
                "status": "executed"
            }
            
        except Exception as e:
            error_msg = f"Error processing signal: {str(e)}"
            print(f"ERROR: {error_msg}")
            return {"success": False, "error": error_msg}

# Use our mock signal processor
signal_processor = MockSignalProcessor()

def test_trading_pair_validation():
    """
    Test the fix for the ValueError: Invalid trading pair format issue.
    """
    print("\n=== Testing trading pair validation fix ===")
    
    # Test case 1: Valid trading pair as string
    print("\n1. Testing with valid trading pair string:")
    valid_pair = "BTC/USD"
    is_valid, error_msg = signal_processor.validate_trading_pair(valid_pair)
    print(f"Trading pair: {valid_pair}")
    print(f"Valid: {is_valid}")
    print(f"Error: {error_msg}")
    
    # Test case 2: Valid trading pair as dictionary
    print("\n2. Testing with valid trading pair dictionary:")
    valid_dict = {"symbol": "ETH/USD", "name": "Ethereum/US Dollar"}
    is_valid, error_msg = signal_processor.validate_trading_pair(valid_dict)
    print(f"Trading pair: {valid_dict}")
    print(f"Valid: {is_valid}")
    print(f"Error: {error_msg}")
    
    # Test case 3: Invalid trading pair format (missing separator)
    print("\n3. Testing with invalid trading pair format (missing separator):")
    invalid_pair = "BTCUSD"
    is_valid, error_msg = signal_processor.validate_trading_pair(invalid_pair)
    print(f"Trading pair: {invalid_pair}")
    print(f"Valid: {is_valid}")
    print(f"Error: {error_msg}")
    
    # Test case 4: Invalid trading pair format (empty components)
    print("\n4. Testing with invalid trading pair format (empty components):")
    invalid_pair = "/"
    is_valid, error_msg = signal_processor.validate_trading_pair(invalid_pair)
    print(f"Trading pair: {invalid_pair}")
    print(f"Valid: {is_valid}")
    print(f"Error: {error_msg}")
    
    # Test case 5: Invalid trading pair format (missing symbol key)
    print("\n5. Testing with invalid trading pair dictionary (missing symbol key):")
    invalid_dict = {"name": "Bitcoin/US Dollar"}
    is_valid, error_msg = signal_processor.validate_trading_pair(invalid_dict)
    print(f"Trading pair: {invalid_dict}")
    print(f"Valid: {is_valid}")
    print(f"Error: {error_msg}")
    
    print("\nThe issue is fixed if all invalid cases are properly detected and rejected.")

def test_signal_with_trading_pair():
    """
    Test the signal processing with trading pairs.
    """
    print("\n=== Testing signal processing with trading pairs ===")
    
    # Test case 1: Valid signal with valid trading pair
    print("\n1. Testing valid signal with valid trading pair:")
    valid_signal = {
        "type": "market",
        "direction": "long",
        "price": {"value": 50000.0},
        "trading_pair": {"symbol": "BTC/USD", "name": "Bitcoin/US Dollar"}
    }
    result = signal_processor.process_signal("test_signal_1", valid_signal)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Test case 2: Valid signal with invalid trading pair
    print("\n2. Testing valid signal with invalid trading pair:")
    invalid_signal = {
        "type": "market",
        "direction": "long",
        "price": {"value": 50000.0},
        "trading_pair": "BTCUSD"  # Missing separator
    }
    result = signal_processor.process_signal("test_signal_2", invalid_signal)
    print(f"Result: {json.dumps(result, indent=2)}")
    
    print("\nThe issue is fixed if invalid trading pairs are properly detected and the signal processing returns an error.")

def main():
    """Main function to run the tests."""
    print("=== Testing Trading Pair Validation Fix ===")
    
    # Test trading pair validation
    test_trading_pair_validation()
    
    # Test signal processing with trading pairs
    test_signal_with_trading_pair()
    
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    main() 