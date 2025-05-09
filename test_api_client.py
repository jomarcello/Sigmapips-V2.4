#!/usr/bin/env python3
"""
Test script voor de TradingAPIClient klasse.
"""

import os
import sys
import json
import logging
import unittest
from unittest.mock import patch, MagicMock, Mock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the TradingAPIClient
from trading_bot.api_client import TradingAPIClient
from utils.api_connector import APIConnector

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

class TestTradingAPIClient(unittest.TestCase):
    """Test cases voor de TradingAPIClient klasse."""
    
    def setUp(self):
        """Set up voor elke test."""
        # Mock de APIConnector klasse
        self.mock_connector = MagicMock(spec=APIConnector)
        
        # Patch de APIConnector constructor
        patcher = patch('trading_bot.api_client.APIConnector', return_value=self.mock_connector)
        self.addCleanup(patcher.stop)
        self.mock_api_connector_class = patcher.start()
        
        # Maak een TradingAPIClient instantie
        self.api_client = TradingAPIClient(
            api_key="test_key",
            api_secret="test_secret",
            base_url="https://api.example.com",
            logger=logger
        )
    
    def test_initialization(self):
        """Test de initialisatie van de TradingAPIClient."""
        # Check of APIConnector correct is aangeroepen
        self.mock_api_connector_class.assert_called_once()
        
        # Check de parameters
        call_kwargs = self.mock_api_connector_class.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://api.example.com")
        self.assertEqual(call_kwargs["headers"]["X-API-Key"], "test_key")
        self.assertEqual(call_kwargs["headers"]["Content-Type"], "application/json")
    
    def test_get_account_info(self):
        """Test het ophalen van account informatie."""
        # Mock de response
        expected_result = {
            "success": True,
            "account_id": "12345",
            "balance": {"BTC": 1.5, "USD": 10000}
        }
        mock_response = MockResponse(json_data=expected_result)
        self.mock_connector.get.return_value = mock_response
        
        # Voer de functie uit
        result = self.api_client.get_account_info()
        
        # Controleer de aanroep en het resultaat
        self.mock_connector.get.assert_called_once_with("account")
        self.assertEqual(result, expected_result)
    
    def test_get_market_price(self):
        """Test het ophalen van marktprijzen."""
        # Mock de response
        expected_result = {
            "symbol": "BTCUSD",
            "price": 55000.50,
            "timestamp": 1622548800
        }
        mock_response = MockResponse(json_data=expected_result)
        self.mock_connector.get.return_value = mock_response
        
        # Voer de functie uit
        result = self.api_client.get_market_price("BTC/USD")
        
        # Controleer de aanroep en het resultaat
        self.mock_connector.get.assert_called_once_with(
            "ticker/BTCUSD",
            params={"precision": "full"}
        )
        self.assertEqual(result, expected_result)
    
    def test_create_order(self):
        """Test het aanmaken van een order."""
        # Mock de response
        expected_result = {
            "order_id": "ord123456",
            "symbol": "BTCUSD",
            "side": "buy",
            "type": "limit",
            "quantity": 0.1,
            "price": 50000.0,
            "status": "new"
        }
        mock_response = MockResponse(json_data=expected_result)
        self.mock_connector.post.return_value = mock_response
        
        # Voer de functie uit
        result = self.api_client.create_order(
            symbol="BTC/USD",
            side="buy",
            order_type="limit",
            quantity=0.1,
            price=50000.0
        )
        
        # Controleer de aanroep en het resultaat
        self.mock_connector.post.assert_called_once()
        call_args, call_kwargs = self.mock_connector.post.call_args
        self.assertEqual(call_args[0], "orders")
        self.assertEqual(call_kwargs["json_data"]["symbol"], "BTCUSD")
        self.assertEqual(call_kwargs["json_data"]["side"], "buy")
        self.assertEqual(call_kwargs["json_data"]["type"], "limit")
        self.assertEqual(call_kwargs["json_data"]["quantity"], 0.1)
        self.assertEqual(call_kwargs["json_data"]["price"], 50000.0)
        self.assertEqual(result, expected_result)
    
    def test_error_handling(self):
        """Test error handling."""
        # Mock een fout
        self.mock_connector.get.side_effect = Exception("Connection refused")
        
        # Voer de functie uit
        result = self.api_client.get_account_info()
        
        # Controleer het resultaat
        self.assertEqual(result["success"], False)
        self.assertEqual(result["error"], "Connection refused")
    
    def test_close(self):
        """Test het sluiten van de verbinding."""
        # Voer de functie uit
        self.api_client.close()
        
        # Controleer of close is aangeroepen op de connector
        self.mock_connector.close.assert_called_once()

def simulate_trading():
    """
    Simuleer een eenvoudige trading workflow.
    """
    print("\n=== Simulating Trading Workflow ===")
    
    # Gebruik de TradingAPIClient als een context manager
    with TradingAPIClient(
        api_key="demo_key",
        api_secret="demo_secret",
        base_url="https://api.example.com"
    ) as client:
        # Mock responses voor een gesimuleerde trading workflow
        with patch.object(client.connector, 'get') as mock_get, \
            patch.object(client.connector, 'post') as mock_post, \
            patch.object(client.connector, 'delete') as mock_delete:
            
            # Mock account info response
            mock_get.return_value = MockResponse(json_data={
                "account_id": "demo123",
                "balance": {"BTC": 0.5, "USD": 25000},
                "status": "active"
            })
            
            # Stap 1: Account info opvragen
            print("\n1. Fetching account information")
            account_info = client.get_account_info()
            print(f"Account info: {json.dumps(account_info, indent=2)}")
            
            # Mock market price response
            mock_get.return_value = MockResponse(json_data={
                "symbol": "BTCUSD",
                "price": 50000.0,
                "bid": 49995.0,
                "ask": 50005.0
            })
            
            # Stap 2: Marktprijs opvragen
            print("\n2. Fetching current BTC/USD price")
            market_price = client.get_market_price("BTC/USD")
            print(f"Market price: {json.dumps(market_price, indent=2)}")
            
            # Mock order creation response
            mock_post.return_value = MockResponse(json_data={
                "order_id": "ord789012",
                "symbol": "BTCUSD",
                "side": "buy",
                "type": "limit",
                "quantity": 0.1,
                "price": 49800.0,
                "status": "new",
                "created_at": "2023-05-15T10:30:00Z"
            })
            
            # Stap 3: Order plaatsen
            print("\n3. Placing a buy limit order")
            buy_order = client.create_order(
                symbol="BTC/USD",
                side="buy",
                order_type="limit",
                quantity=0.1,
                price=49800.0
            )
            print(f"Buy order: {json.dumps(buy_order, indent=2)}")
            
            # Mock order status response
            mock_get.return_value = MockResponse(json_data={
                "order_id": "ord789012",
                "symbol": "BTCUSD",
                "side": "buy",
                "type": "limit",
                "quantity": 0.1,
                "price": 49800.0,
                "status": "filled",
                "filled_quantity": 0.1,
                "filled_price": 49800.0,
                "created_at": "2023-05-15T10:30:00Z",
                "updated_at": "2023-05-15T10:35:00Z"
            })
            
            # Stap 4: Order status controleren
            print("\n4. Checking order status")
            order_status = client.get_order_status("ord789012")
            print(f"Order status: {json.dumps(order_status, indent=2)}")
            
            # Mock cancel order response
            mock_delete.return_value = MockResponse(json_data={
                "order_id": "ord789012",
                "success": True,
                "message": "Order successfully cancelled"
            })
            
            # Stap 5: Order annuleren (als voorbeeld)
            print("\n5. Cancelling the order (example only)")
            cancel_result = client.cancel_order("ord789012")
            print(f"Cancel result: {json.dumps(cancel_result, indent=2)}")

def main():
    """Main function."""
    # Run unittest tests
    unittest.main(exit=False)
    
    # Run simulation
    simulate_trading()

if __name__ == "__main__":
    main() 