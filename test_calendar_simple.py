import asyncio
import os
import logging
import json
import aiohttp
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class SimpleTradingViewCalendarTester:
    """Simplified TradingView calendar tester without dependencies"""
    
    def __init__(self):
        self.base_url = "https://economic-calendar.tradingview.com/events"
        self.session = None
        self.use_scrapingant = True
        self.scrapingant_api_key = "e63e79e708d247c798885c0c320f9f30"
        self.scrapingant_url = "https://api.scrapingant.com/v2/general"
    
    async def _ensure_session(self):
        """Ensure we have an active aiohttp session"""
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def _close_session(self):
        """Close the aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _format_date(self, date: datetime) -> str:
        """Format date for TradingView API"""
        date = date.replace(microsecond=0)
        return date.isoformat() + '.000Z'
    
    async def test_direct_api(self):
        """Test direct TradingView API connection"""
        try:
            logger.info("Testing direct API connection to TradingView")
            await self._ensure_session()
            
            # Prepare parameters
            start_date = datetime.now()
            end_date = start_date + timedelta(days=1)
            
            params = {
                'from': self._format_date(start_date),
                'to': self._format_date(end_date),
                'countries': 'US,EU,GB,JP,CH,AU,NZ,CA',
                'limit': 10
            }
            
            # Add headers for better API compatibility
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Origin": "https://www.tradingview.com",
                "Referer": "https://www.tradingview.com/economic-calendar/",
                "Connection": "keep-alive",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache"
            }
            
            # Make request
            full_url = f"{self.base_url}"
            logger.info(f"Making request to: {full_url}")
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with self.session.get(full_url, params=params, headers=headers, timeout=timeout) as response:
                logger.info(f"Response status: {response.status}")
                
                if response.status == 200:
                    response_text = await response.text()
                    logger.info(f"Response preview: {response_text[:200]}...")
                    
                    # Check if it's valid JSON
                    if response_text.strip().startswith('[') or response_text.strip().startswith('{'):
                        logger.info("Response appears to be valid JSON")
                        # Try to parse a few items
                        try:
                            data = json.loads(response_text)
                            if isinstance(data, list):
                                logger.info(f"Got {len(data)} events in list format")
                                if len(data) > 0:
                                    logger.info(f"First event: {json.dumps(data[0])}")
                            elif isinstance(data, dict):
                                logger.info(f"Got response in dict format with keys: {list(data.keys())}")
                                # Try to extract result list if available
                                if "result" in data and isinstance(data["result"], list):
                                    logger.info(f"Found {len(data['result'])} events in result list")
                                    if len(data["result"]) > 0:
                                        logger.info(f"First event: {json.dumps(data['result'][0])}")
                            return True
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON: {str(e)}")
                    else:
                        logger.error("Response is not valid JSON")
                else:
                    logger.error(f"Error response from API: {response.status}")
                
            return False
        except Exception as e:
            logger.error(f"Error during direct API test: {str(e)}")
            return False
    
    async def test_scrapingant(self):
        """Test ScrapingAnt API connection"""
        try:
            logger.info("Testing ScrapingAnt API connection")
            await self._ensure_session()
            
            # Prepare parameters
            start_date = datetime.now()
            end_date = start_date + timedelta(days=1)
            
            params = {
                'from': self._format_date(start_date),
                'to': self._format_date(end_date),
                'countries': 'US,EU,GB,JP,CH,AU,NZ,CA',
                'limit': 10
            }
            
            # Build full URL with query parameters
            import urllib.parse
            query_string = urllib.parse.urlencode(params)
            full_url = f"{self.base_url}?{query_string}"
            
            # Maak de URL's URL encoded
            encoded_url = urllib.parse.quote(full_url, safe='')
            logger.info(f"Encoded URL: {encoded_url}")
            
            # Bereid aanvraag parameters voor volgens ScrapingAnt documentatie
            request_body = {
                "url": full_url
            }
            
            logger.info(f"Making ScrapingAnt request for URL: {full_url}")
            logger.info(f"Using ScrapingAnt API key: {self.scrapingant_api_key[:5]}...{self.scrapingant_api_key[-3:]}")
            
            # Voeg headers toe voor de API key
            headers = {
                "x-api-key": self.scrapingant_api_key
            }
            
            # Make request to ScrapingAnt
            timeout = aiohttp.ClientTimeout(total=60)
            async with self.session.post(
                self.scrapingant_url,
                json=request_body,
                headers=headers,
                timeout=timeout
            ) as response:
                logger.info(f"ScrapingAnt response status: {response.status}")
                
                if response.status == 200:
                    response_data = await response.json()
                    logger.info(f"ScrapingAnt response keys: {list(response_data.keys())}")
                    
                    # Extract content
                    content = None
                    if "text" in response_data:
                        content = response_data["text"]
                        logger.info("Got text content from ScrapingAnt")
                    elif "html" in response_data:
                        content = response_data["html"]
                        logger.info("Got HTML content from ScrapingAnt")
                    elif "content" in response_data:
                        content = response_data["content"]
                        logger.info("Got content from ScrapingAnt")
                    
                    if content:
                        logger.info(f"Content preview: {content[:200]}...")
                        
                        # Check if it looks like JSON
                        if content.strip().startswith('[') or content.strip().startswith('{'):
                            logger.info("Content appears to be valid JSON")
                            # Try to parse
                            try:
                                data = json.loads(content)
                                if isinstance(data, list):
                                    logger.info(f"Got {len(data)} events in list format")
                                    if len(data) > 0:
                                        logger.info(f"First event: {json.dumps(data[0])}")
                                elif isinstance(data, dict):
                                    logger.info(f"Got response in dict format with keys: {list(data.keys())}")
                                    # Try to extract result list if available
                                    if "result" in data and isinstance(data["result"], list):
                                        logger.info(f"Found {len(data['result'])} events in result list")
                                        if len(data["result"]) > 0:
                                            logger.info(f"First event: {json.dumps(data['result'][0])}")
                                return True
                            except json.JSONDecodeError as e:
                                logger.error(f"Invalid JSON: {str(e)}")
                        else:
                            logger.error("Content is not valid JSON")
                else:
                    logger.error(f"Error response from ScrapingAnt: {response.status}")
                    try:
                        error_text = await response.text()
                        logger.error(f"Error details: {error_text[:200]}")
                    except:
                        pass
            
            return False
        except Exception as e:
            logger.error(f"Error during ScrapingAnt test: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

async def main():
    """Run tests"""
    logger.info("Starting TradingView Calendar API tests")
    
    tester = SimpleTradingViewCalendarTester()
    
    try:
        # Test direct API
        logger.info("=== Testing Direct API ===")
        direct_api_success = await tester.test_direct_api()
        
        # Test ScrapingAnt API
        logger.info("=== Testing ScrapingAnt API ===")
        scrapingant_success = await tester.test_scrapingant()
        
        # Log results
        logger.info("=== Test Results ===")
        logger.info(f"Direct API: {'SUCCESS' if direct_api_success else 'FAILED'}")
        logger.info(f"ScrapingAnt API: {'SUCCESS' if scrapingant_success else 'FAILED'}")
    
    except Exception as e:
        logger.error(f"Error during tests: {str(e)}")
    
    finally:
        await tester._close_session()
    
    logger.info("Tests completed")

if __name__ == "__main__":
    asyncio.run(main()) 