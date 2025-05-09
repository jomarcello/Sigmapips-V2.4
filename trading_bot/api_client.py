"""
API Client voor de trading bot.
Gebruikt de APIConnector voor robuuste verbindingen met externe APIs.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union

from utils.api_connector import APIConnector

logger = logging.getLogger(__name__)

class TradingAPIClient:
    """
    Client voor interactie met de externe trading APIs.
    Gebruikt de APIConnector voor robuuste verbindingen.
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 5,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialiseert de trading API client.
        
        Args:
            api_key: API key voor authenticatie
            api_secret: API secret voor authenticatie
            base_url: Basis URL voor de API
            max_retries: Maximum aantal retries bij verbindingsproblemen
            logger: Logger instance om berichten naar te loggen
        """
        self.api_key = api_key or os.environ.get("TRADING_API_KEY")
        self.api_secret = api_secret or os.environ.get("TRADING_API_SECRET")
        self.base_url = base_url or os.environ.get("TRADING_API_URL", "https://api.example.com")
        self.logger = logger or logging.getLogger(__name__)
        
        # Basisheaders instellen
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Sigmapips Trading Bot/5.4",
        }
        
        # API key toevoegen aan headers indien aanwezig
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        # Connector instellen met retry functionaliteit
        self.connector = APIConnector(
            base_url=self.base_url,
            max_retries=max_retries,
            headers=headers,
            logger=self.logger
        )
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Haalt account informatie op van de trading API.
        
        Returns:
            Dict met account informatie
        """
        self.logger.info("Fetching account information")
        
        try:
            response = self.connector.get("account")
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get account info: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def get_market_price(self, symbol: str) -> Dict[str, Any]:
        """
        Haalt de huidige marktprijs op voor een specifiek trading pair.
        
        Args:
            symbol: Trading pair symbool (bijv. "BTC/USD")
            
        Returns:
            Dict met prijsinformatie
        """
        self.logger.info(f"Fetching market price for {symbol}")
        
        # Converteer het trading pair naar het juiste format voor de API
        # Sommige APIs gebruiken BTC-USD, andere BTCUSD, etc.
        formatted_symbol = symbol.replace("/", "")
        
        try:
            response = self.connector.get(
                f"ticker/{formatted_symbol}",
                params={"precision": "full"}
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get market price for {symbol}: {str(e)}")
            return {"success": False, "error": str(e), "symbol": symbol}
    
    def create_order(
        self, 
        symbol: str, 
        side: str, 
        order_type: str, 
        quantity: float, 
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Maakt een nieuwe order aan.
        
        Args:
            symbol: Trading pair symbool (bijv. "BTC/USD")
            side: "buy" of "sell"
            order_type: "market" of "limit"
            quantity: Hoeveelheid om te kopen/verkopen
            price: Prijs voor limit orders (None voor market orders)
            
        Returns:
            Dict met orderinformatie
        """
        self.logger.info(f"Creating {order_type} order: {side} {quantity} {symbol}")
        
        # Bouw de order data
        order_data = {
            "symbol": symbol.replace("/", ""),
            "side": side,
            "type": order_type,
            "quantity": quantity
        }
        
        # Voeg prijs toe voor limit orders
        if order_type == "limit" and price is not None:
            order_data["price"] = price
        
        try:
            response = self.connector.post(
                "orders",
                json_data=order_data
            )
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to create order: {str(e)}")
            return {"success": False, "error": str(e), "order": order_data}
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Haalt de status van een specifieke order op.
        
        Args:
            order_id: ID van de order
            
        Returns:
            Dict met order status
        """
        self.logger.info(f"Fetching status for order {order_id}")
        
        try:
            response = self.connector.get(f"orders/{order_id}")
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get order status for {order_id}: {str(e)}")
            return {"success": False, "error": str(e), "order_id": order_id}
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Annuleert een specifieke order.
        
        Args:
            order_id: ID van de order
            
        Returns:
            Dict met resultaat van annulering
        """
        self.logger.info(f"Cancelling order {order_id}")
        
        try:
            response = self.connector.delete(f"orders/{order_id}")
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return {"success": False, "error": str(e), "order_id": order_id}
    
    def get_open_orders(self) -> List[Dict[str, Any]]:
        """
        Haalt alle open orders op.
        
        Returns:
            Lijst met open orders
        """
        self.logger.info("Fetching open orders")
        
        try:
            response = self.connector.get("orders", params={"status": "open"})
            return response.json()
        except Exception as e:
            self.logger.error(f"Failed to get open orders: {str(e)}")
            return [{"success": False, "error": str(e)}]
    
    def close(self) -> None:
        """Sluit de API verbinding."""
        self.connector.close()
    
    def __enter__(self) -> 'TradingAPIClient':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit, sluit de verbinding."""
        self.close() 