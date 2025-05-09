"""
Authentication module for the trading bot.
"""
import os
import json
import requests
from datetime import datetime, timedelta

from trading_bot.utils.debug_logger import log_input, log_output, log_error, log_variable, log_process

class AuthManager:
    """
    Manages user authentication and token handling.
    """
    
    def __init__(self, api_url=None):
        """Initialize the auth manager."""
        self.api_url = api_url or os.environ.get("API_URL", "https://api.example.com")
        self.tokens = {}
    
    def authenticate_user(self, username, password):
        """
        Authenticate a user with username and password.
        
        Args:
            username (str): User's username
            password (str): User's password
            
        Returns:
            dict: Authentication result with token
        """
        # Log input data (masking password for security)
        log_input({"username": username, "password": "********"}, source="authenticate_user")
        log_process("user_authentication", {"username": username}, "started")
        
        try:
            # Simulate API call
            log_variable("auth_api_url", f"{self.api_url}/auth")
            
            # In a real implementation, this would be an actual API call
            # response = requests.post(f"{self.api_url}/auth", json={"username": username, "password": password})
            # user_data = response.json()
            
            # Simulated response
            user_data = self._simulate_auth_response(username, password)
            log_variable("user_data", {k: v for k, v in user_data.items() if k != "password"})
            
            # This is the line mentioned in the error logs
            # We add validation to prevent the KeyError
            if "auth_token" not in user_data:
                error_msg = "Authentication failed: Missing auth_token in response"
                log_error(error_msg, {"username": username})
                log_process("user_authentication", {"username": username}, "failed")
                return {"success": False, "error": error_msg}
            
            # Extract token from response
            token = user_data["auth_token"]
            log_variable("token_received", bool(token))
            
            # Store token
            self.tokens[username] = {
                "token": token,
                "expires_at": datetime.now() + timedelta(hours=24)
            }
            
            result = {
                "success": True,
                "username": username,
                "token": token,
                "expires_at": (datetime.now() + timedelta(hours=24)).isoformat()
            }
            
            log_output(result, "authenticate_user")
            log_process("user_authentication", {"username": username}, "completed")
            return result
            
        except Exception as e:
            log_error(e, {"username": username, "function": "authenticate_user"})
            log_process("user_authentication", {"username": username}, "failed")
            return {"success": False, "error": str(e)}
    
    def _simulate_auth_response(self, username, password):
        """Simulate authentication API response for demonstration."""
        if username == "test_user" and password == "correct_password":
            return {
                "user_id": 12345,
                "username": username,
                "auth_token": "simulated_token_12345",
                "role": "user"
            }
        elif username == "error_user":
            # Simulate the error from the logs - missing auth_token
            return {
                "user_id": 67890,
                "username": username,
                "role": "user"
                # auth_token is missing intentionally
            }
        else:
            return {
                "error": "Invalid credentials"
            }
    
    def validate_token(self, token):
        """
        Validate if a token is valid and not expired.
        
        Args:
            token (str): Authentication token
            
        Returns:
            bool: True if token is valid, False otherwise
        """
        log_input({"token": f"{token[:5]}..."}, source="validate_token")
        log_process("token_validation", {"token_prefix": token[:5]}, "started")
        
        try:
            # Check if token exists in our stored tokens
            for username, token_data in self.tokens.items():
                if token_data["token"] == token:
                    # Check if token is expired
                    if datetime.now() < token_data["expires_at"]:
                        log_output({"valid": True, "username": username}, "validate_token")
                        log_process("token_validation", {"token_prefix": token[:5]}, "completed")
                        return True
            
            log_output({"valid": False}, "validate_token")
            log_process("token_validation", {"token_prefix": token[:5]}, "failed")
            return False
            
        except Exception as e:
            log_error(e, {"token_prefix": token[:5], "function": "validate_token"})
            log_process("token_validation", {"token_prefix": token[:5]}, "failed")
            return False

# Create a default instance
auth_manager = AuthManager() 