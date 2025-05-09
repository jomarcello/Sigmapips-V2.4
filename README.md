# Sigmapips AI-Powered Debug Toolkit

A comprehensive debugging toolkit for the Sigmapips trading bot, using AI to streamline the debugging process.

## Overview

This toolkit provides a systematic approach to debugging the Sigmapips trading bot by:

1. **Capturing Detailed Logs**: Rotating logs with comprehensive information about inputs, outputs, errors, and application state
2. **Adding Targeted Logging**: Automatically add logging to critical areas of the codebase
3. **Analyzing Logs with AI**: Process logs to identify patterns and suggest fixes

## Components

### Debug Logger
`trading_bot/utils/debug_logger.py` - A rotating log system that captures:
- Function inputs and outputs
- Process status (started, completed, failed)
- Errors with full tracebacks
- Variable values

```python
# Example usage
from trading_bot.utils.debug_logger import log_input, log_output, log_error, log_variable, log_process

# Log inputs to a function
log_input({"user_id": 123, "action": "login"}, source="authenticate_user")

# Log process status
log_process("authentication", {"user_id": 123}, "started")

# Log variable values
log_variable("auth_result", result)

# Log errors
try:
    # code that might raise an exception
    result = some_function()
except Exception as e:
    log_error(e, {"function": "authenticate_user", "user_id": 123})
    
# Log process completion
log_process("authentication", {"user_id": 123}, "completed")

# Log outputs from a function
log_output({"success": True, "token": "abc123"}, "authenticate_user")
```

### Railway Log Analyzer
`utils/railway_log_analyzer.py` - A tool to fetch and analyze logs from Railway:
- Extracts error patterns from logs
- Identifies relevant files and functions
- Generates AI-friendly analysis reports

### Debug Integrator
`utils/debug_integrator.py` - A utility to automatically add logging to existing code:
- Analyzes code to find critical sections
- Adds appropriate logging statements
- Preserves code functionality

### API Connector
`utils/api_connector.py` - A robust connector for API communication:
- Implements automatic retry with exponential backoff
- Adds jitter for optimal reconnection timing
- Provides comprehensive logging
- Handles errors gracefully

```python
# Example usage
from utils.api_connector import APIConnector

# Create a connector
connector = APIConnector(
    base_url="https://api.example.com",
    max_retries=5,
    backoff_factor=0.5,
    timeout=10
)

# Make a request with automatic retry
try:
    response = connector.get("users/123")
    user_data = response.json()
except Exception as e:
    print(f"Failed after multiple retries: {e}")
```

### Trading API Client
`trading_bot/api_client.py` - A high-level client for trading APIs:
- Uses the APIConnector for robust connections
- Provides clean interfaces for common trading operations
- Handles errors consistently
- Supports environment variables for configuration

```python
# Example usage
from trading_bot.api_client import TradingAPIClient

# Create a client
client = TradingAPIClient(
    api_key="your_api_key",
    api_secret="your_api_secret"
)

# Get account information
account_info = client.get_account_info()

# Place an order
order = client.create_order(
    symbol="BTC/USD",
    side="buy",
    order_type="limit",
    quantity=0.1,
    price=50000.0
)
```

## Debugging and Testing

### Example Debug Standalone
`example_debug_standalone.py` - A standalone simulation to demonstrate the debugging workflow:
- Simulates common errors in the trading bot
- Demonstrates how the debug toolkit identifies and logs issues

### Testing Tools
- `test_api_connector.py` - Tests for the APIConnector
- `test_api_client.py` - Tests for the TradingAPIClient
- `test_trading_pair_fix.py` - Tests for the trading pair validation fix

## Documentation

- `AI_DEBUGGING_SYSTEM.md` - Comprehensive documentation for the AI-powered debugging system
- `DEBUGGING_RESULTS.md` - Results from the debug sessions
- `IMPROVEMENTS.md` - Overview of improvements made to the codebase

## Getting Started

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run a test simulation**:
```bash
python example_debug_standalone.py
```

3. **Analyze real logs**:
```bash
python debug_railway_with_ai.py debug "Authentication and signal processing errors"
```

4. **View the analysis**:
```bash
cat railway_logs/ai_analysis_*.json
```

## Major Bug Fixes

This toolkit has already successfully identified and fixed several critical bugs:

1. **KeyError: 'auth_token'** in authenticate_user function
2. **TypeError: 'NoneType' object is not subscriptable** in process_signal function
3. **ValueError: Invalid trading pair format** in validate_signal function
4. **ConnectionError: Failed to connect to trading API**

See `DEBUGGING_RESULTS.md` for detailed descriptions of these fixes.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
