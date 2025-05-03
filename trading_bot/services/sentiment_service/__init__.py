# Sentiment service initialization

# Define MarketSentimentService class directly in __init__.py to avoid circular imports
import logging
import random
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Simple cache for sentiment data
_sentiment_cache = {}

class MarketSentimentService:
    """Service for analyzing market sentiment (simplified version in __init__.py)"""
    
    def __init__(self):
        """Initialize the market sentiment service"""
        self.use_mock = True
        self.cache_ttl = 30 * 60
        logger.info("MarketSentimentService initialized with cache TTL: %s seconds", self.cache_ttl)
    
    async def get_market_sentiment(self, instrument):
        """Get market sentiment for a specific instrument"""
        logger.info(f"Getting market sentiment for {instrument}")
        return await self._generate_mock_data(instrument)
                
    async def _generate_mock_data(self, instrument):
        """Returns an error message instead of generating mock data"""
        logger.warning(f"Mock data requested for {instrument} but generation is disabled")
        
        # Error message instead of mock data
        result = {
            'overall_sentiment': 'unknown',
            'sentiment_score': 0,
            'bullish_percentage': 0,
            'analysis': f"<b>⚠️ Sentiment Analysis Unavailable</b>\n\nNo real sentiment data is available for {instrument}. The system does not use fallback data.",
            'source': 'error'
        }
        
        # Save to cache
        _sentiment_cache[f"market_{instrument}"] = {
            'result': result,
            'timestamp': datetime.now()
        }
        
        return result
    
    async def get_sentiment(self, instrument, market_type=None):
        """Get sentiment for a given instrument"""
        return await self.get_market_sentiment(instrument)
        
    async def get_market_sentiment_html(self, instrument):
        """Get HTML-formatted sentiment analysis"""
        sentiment_data = await self.get_market_sentiment(instrument)
        return sentiment_data.get('analysis', f"<b>No sentiment data available for {instrument}</b>")

# Export the class
__all__ = ["MarketSentimentService"]
