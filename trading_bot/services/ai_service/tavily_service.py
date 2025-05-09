import json
import os
import httpx
from typing import Optional, Dict, Any, List
import asyncio
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class TavilyService:
    def __init__(self, api_key: Optional[str] = None, api_timeout: int = 30, metrics=None):
        """Initialize the Tavily service with API key"""
        # Store parameters for compatibility but don't actually use them
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
        self.api_timeout = api_timeout
        self.metrics = metrics
        logger.info("Initializing TavilyService in mock mode (Tavily not used)")
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Return mock search results instead of querying Tavily API"""
        logger.info(f"Mock Tavily search for: {query}")
        
        # Generate mock search results based on query keywords
        results = []
        
        if "economic calendar" in query.lower() or "calendar" in query.lower():
            results.append({
                "title": "Economic Calendar - Today's Economic Events",
                "url": "https://www.forexfactory.com/calendar",
                "content": "Economic calendar showing major events for today. The calendar includes data for USD, EUR, GBP, JPY, AUD, CAD, CHF, and NZD currencies. Upcoming events include interest rate decisions, employment reports, and inflation data. Each event is marked with an impact level (high, medium, or low)."
            })
            
        # Always include at least one result regardless of query
        if not results:
            results.append({
                "title": "Financial Markets Overview",
                "url": "https://www.bloomberg.com/markets",
                "content": f"Latest market data and analysis as of {datetime.now().strftime('%B %d, %Y')}. Market sentiment is mixed with various economic indicators showing conflicting signals. Recent economic data shows moderate growth with inflation concerns easing."
            })
            
        logger.info(f"Returning {len(results)} mock results")
        return results
            
    async def search_internet(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Return mock internet search results"""
        results = await self.search(query, max_results)
        return {"results": results}
