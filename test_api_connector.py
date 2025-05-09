#!/usr/bin/env python3
"""
Test script voor de APIConnector klasse om het retry mechanisme te verifiëren.
"""

import os
import sys
import time
import logging
import unittest
from unittest.mock import patch, MagicMock, call

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the APIConnector
from utils.api_connector import APIConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockResponse:
    """Mock voor requests.Response object."""
    
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text
    
    def json(self):
        return self._json_data
    
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP Error: {self.status_code}")

class TestAPIConnector(unittest.TestCase):
    """Test cases voor de APIConnector klasse."""
    
    def setUp(self):
        """Set up voor elke test."""
        self.base_url = "https://api.example.com"
        self.api = APIConnector(
            base_url=self.base_url,
            max_retries=3,
            backoff_factor=0.01,  # Maak dit klein voor snelle tests
            timeout=5,
            logger=logger
        )
        
        # Override de _calculate_backoff methode om tests sneller te maken
        self.original_calculate_backoff = self.api._calculate_backoff
        self.api._calculate_backoff = lambda attempt: 0.001  # Zeer korte wachttijd voor tests
    
    def tearDown(self):
        """Clean up na elke test."""
        # Herstel de originele methode
        self.api._calculate_backoff = self.original_calculate_backoff
        self.api.close()
    
    @patch('requests.Session.get')
    def test_successful_request(self, mock_get):
        """Test een succesvolle API request."""
        # Mock de response
        mock_response = MockResponse(
            status_code=200,
            json_data={"success": True, "data": {"message": "Success"}}
        )
        mock_get.return_value = mock_response
        
        # Voer de request uit
        response = self.api.get("users")
        
        # Controleer of de juiste URL is aangeroepen
        mock_get.assert_called_once()
        call_args = mock_get.call_args[0][0]
        self.assertEqual(call_args, f"{self.base_url}/users")
        
        # Controleer de response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"success": True, "data": {"message": "Success"}})
    
    def test_retry_mechanism(self):
        """Test het retry mechanisme bij falen met een custom functie."""
        # Maak een mock functie die de eerste twee keer faalt en daarna slaagt
        mock_func = MagicMock()
        mock_func.side_effect = [
            Exception("First failure"),
            Exception("Second failure"),
            "Success on third try"
        ]
        
        # Voer het retry mechanisme direct uit
        result = self.api.with_retry(mock_func, "test_arg", test_kwarg=123)
        
        # Controleer of de functie 3 keer is aangeroepen
        self.assertEqual(mock_func.call_count, 3)
        
        # Controleer of de functie met de juiste argumenten is aangeroepen
        mock_func.assert_has_calls([
            call("test_arg", test_kwarg=123),
            call("test_arg", test_kwarg=123),
            call("test_arg", test_kwarg=123)
        ])
        
        # Controleer het resultaat
        self.assertEqual(result, "Success on third try")
    
    def test_max_retries_exceeded(self):
        """Test dat een exception wordt geraised als max_retries wordt overschreden."""
        # Maak een mock functie die altijd faalt
        test_exception = Exception("Test failure")
        mock_func = MagicMock(side_effect=test_exception)
        
        # Controleer of de juiste exception wordt geraised
        with self.assertRaises(Exception) as context:
            self.api.with_retry(mock_func)
        
        # Controleer het aantal pogingen
        expected_calls = self.api.max_retries + 1  # Initiële poging + retries
        self.assertEqual(mock_func.call_count, expected_calls)
        
        # Controleer de foutmelding
        self.assertEqual(str(context.exception), "Test failure")
    
    @patch('requests.Session.post')
    def test_post_request(self, mock_post):
        """Test een POST request."""
        # Mock de response
        mock_response = MockResponse(
            status_code=201,
            json_data={"success": True, "data": {"id": 123}}
        )
        mock_post.return_value = mock_response
        
        # Testdata
        data = {"name": "Test User", "email": "test@example.com"}
        
        # Voer de request uit
        response = self.api.post("users", json_data=data)
        
        # Controleer of de juiste URL en data zijn gebruikt
        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        self.assertEqual(call_args[0], f"{self.base_url}/users")
        self.assertEqual(call_kwargs["json"], data)
        
        # Controleer de response
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json(), {"success": True, "data": {"id": 123}})

def main():
    """Run de tests."""
    unittest.main()

if __name__ == "__main__":
    main() 