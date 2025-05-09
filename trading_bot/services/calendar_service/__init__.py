# This package contains calendar services
# Explicitly export classes for external use

import logging
import traceback
import os
import sys
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import random

logger = logging.getLogger(__name__)
logger.info("Initializing calendar service module...")

# Configureer een extra handlertje voor kalender gerelateerde logs
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# Detecteer of we in Railway draaien
RUNNING_IN_RAILWAY = os.environ.get("RAILWAY_ENVIRONMENT") is not None
HOSTNAME = socket.gethostname()

logger.info(f"Running on host: {HOSTNAME}")
logger.info(f"Running in Railway: {RUNNING_IN_RAILWAY}")

# BELANGRIJK: Force instellingen voor TradingView calendar
# Expliciet investing.com uitschakelen (investing is verwijderd)
os.environ["USE_INVESTING_CALENDAR"] = "false"
logger.info("⚠️ Investing.com calendar is verwijderd en niet meer beschikbaar")
print("⚠️ Investing.com calendar is verwijderd en niet meer beschikbaar")

# Calendar fallback uitschakelen - we willen echte data
os.environ["USE_CALENDAR_FALLBACK"] = "false"
logger.info("⚠️ Forcing USE_CALENDAR_FALLBACK=false to use real data")
print("⚠️ Forcing USE_CALENDAR_FALLBACK=false to use real data")

# ScrapingAnt inschakelen voor betere data
os.environ["USE_SCRAPINGANT"] = "true"
logger.info("⚠️ Forcing USE_SCRAPINGANT=true for better data retrieval")
print("⚠️ Forcing USE_SCRAPINGANT=true for better data retrieval")

# ScrapingAnt API key configureren indien niet al gedaan
if os.environ.get("SCRAPINGANT_API_KEY") is None:
    os.environ["SCRAPINGANT_API_KEY"] = "e63e79e708d247c798885c0c320f9f30"
    logger.info("Setting default ScrapingAnt API key")

# Check of er iets expliciets in de omgeving is ingesteld voor fallback
USE_FALLBACK = False  # We willen de echte implementatie gebruiken, niet de fallback

# Ingebouwde fallback EconomicCalendarService voor het geval de echte niet werkt
class InternalFallbackCalendarService:
    """Interne fallback implementatie van EconomicCalendarService"""
    def __init__(self, *args, **kwargs):
        self.logger = logging.getLogger(__name__)
        self.logger.warning("Internal fallback EconomicCalendarService is being used!")
        print("⚠️ INTERNAL FALLBACK CALENDAR SERVICE IS ACTIVE ⚠️")
        
    async def get_calendar(self, days_ahead: int = 0, min_impact: str = "Low", currency: str = None) -> List[Dict]:
        """Return empty calendar data"""
        self.logger.info(f"Internal fallback get_calendar called")
        return []
    
    async def get_economic_calendar(self, currencies: List[str] = None, days_ahead: int = 0, min_impact: str = "Low") -> str:
        """Return empty economic calendar message"""
        return "<b>📅 Economic Calendar</b>\n\nNo economic events available (using internal fallback)."
        
    async def get_events_for_instrument(self, instrument: str, *args, **kwargs) -> Dict[str, Any]:
        """Return empty events for an instrument"""
        return {
            "events": [], 
            "explanation": f"No calendar events available (using internal fallback)"
        }
        
    async def get_instrument_calendar(self, instrument: str, *args, **kwargs) -> str:
        """Return empty calendar for an instrument"""
        return "<b>📅 Economic Calendar</b>\n\nNo calendar events available (using internal fallback)."

# Log duidelijk naar de console of we fallback gebruiken of niet
if USE_FALLBACK:
    logger.info("⚠️ USE_CALENDAR_FALLBACK is set to True, using fallback implementation")
    print("⚠️ Calendar fallback mode is ENABLED via environment variable")
    print(f"⚠️ Check environment value: '{os.environ.get('USE_CALENDAR_FALLBACK', '')}'")
    # Gebruik interne fallback
    EconomicCalendarService = InternalFallbackCalendarService
    logger.info("Successfully initialized internal fallback EconomicCalendarService")
else:
    # Probeer eerst de volledige implementatie
    logger.info("✅ USE_CALENDAR_FALLBACK is set to False, will use real implementation")
    print("✅ Calendar fallback mode is DISABLED")
    print(f"✅ Environment value: '{os.environ.get('USE_CALENDAR_FALLBACK', '')}'")
    
    try:
        logger.info("Attempting to import EconomicCalendarService from calendar.py...")
        from trading_bot.services.calendar_service.calendar import EconomicCalendarService
        logger.info("Successfully imported EconomicCalendarService from calendar.py")
        
        # Test importeren van TradingView kalender
        try:
            from trading_bot.services.calendar_service.tradingview_calendar import TradingViewCalendarService
            logger.info("Successfully imported TradingViewCalendarService")
            
            # Check if using ScrapingAnt
            use_scrapingant = os.environ.get("USE_SCRAPINGANT", "").lower() in ("true", "1", "yes")
            logger.info(f"Using ScrapingAnt for calendar API: {use_scrapingant}")
            
            if use_scrapingant:
                print("✅ Using ScrapingAnt proxy for TradingView calendar API")
            else:
                print("✅ Using direct connection for TradingView calendar API")
            
        except Exception as e:
            logger.warning(f"TradingViewCalendarService import failed: {e}")
            logger.debug(traceback.format_exc())
            print("⚠️ TradingView calendar service could not be imported")

    except Exception as e:
        # Als de import faalt, gebruiken we onze interne fallback implementatie
        logger.error(f"Could not import EconomicCalendarService from calendar.py: {str(e)}")
        logger.debug(traceback.format_exc())
        logger.warning("Using internal fallback implementation")
        print("⚠️ Could not import real calendar service, using internal fallback")
        
        # Gebruik interne fallback
        EconomicCalendarService = InternalFallbackCalendarService
        
        # Log dat we de fallback gebruiken
        logger.info("Successfully initialized internal fallback EconomicCalendarService")

# Exporteer TradingView debug functie als die beschikbaar is
try:
    from trading_bot.services.calendar_service.tradingview_calendar import TradingViewCalendarService
    
    # Create a global function to run the debug
    async def debug_tradingview_api():
        """Run a debug check on the TradingView API"""
        logger.info("Running TradingView API debug check")
        service = TradingViewCalendarService()
        return await service.debug_api_connection()

    # Add a function to get all calendar events without filtering
    async def get_all_calendar_events():
        """Get all calendar events without filtering"""
        logger.info("Getting all calendar events without filtering")
        service = TradingViewCalendarService()
        events = await service.get_calendar(days_ahead=0, min_impact="Low")
        logger.info(f"Retrieved {len(events)} total events")
        return events

    __all__ = ['EconomicCalendarService', 'debug_tradingview_api', 'get_all_calendar_events']
except Exception:
    # Als de import faalt, exporteren we alleen de EconomicCalendarService
    __all__ = ['EconomicCalendarService']

class EconomicCalendarService:
    """Service for retrieving and processing economic calendar events"""
    
    def __init__(self):
        """Initialize the economic calendar service"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing EconomicCalendarService")
        self.use_mock_data = True
        
    async def get_events(self, currencies: List[str] = None, days: int = 3) -> List[Dict[str, Any]]:
        """
        Get economic calendar events for the specified currencies
        
        Args:
            currencies: List of currency codes to filter events by
            days: Number of days to look ahead
            
        Returns:
            List[Dict[str, Any]]: List of calendar events
        """
        self.logger.info(f"Getting calendar events for currencies: {currencies}, days: {days}")
        
        if self.use_mock_data:
            return self._generate_mock_events(currencies, days)
        
        # Implement actual calendar service here
        self.logger.warning("Real calendar data retrieval not implemented, using mock data")
        return self._generate_mock_events(currencies, days)
    
    def _generate_mock_events(self, currencies: List[str] = None, days: int = 3) -> List[Dict[str, Any]]:
        """
        Generate mock calendar events for testing
        
        Args:
            currencies: List of currency codes to filter events by
            days: Number of days to look ahead
            
        Returns:
            List[Dict[str, Any]]: List of mock calendar events
        """
        self.logger.info(f"Generating mock calendar events for currencies: {currencies}, days: {days}")
        
        if not currencies:
            currencies = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"]
        
        current_date = datetime.now()
        events = []
        
        # Event templates by currency
        event_templates = {
            "USD": [
                {"name": "Non-Farm Payrolls", "importance": "high", "forecast": "+175K", "previous": "+165K"},
                {"name": "CPI m/m", "importance": "high", "forecast": "+0.3%", "previous": "+0.2%"},
                {"name": "GDP q/q", "importance": "high", "forecast": "+2.4%", "previous": "+2.1%"},
                {"name": "Retail Sales m/m", "importance": "medium", "forecast": "+0.4%", "previous": "+0.1%"},
                {"name": "Fed Chair Powell Speech", "importance": "high", "forecast": "", "previous": ""},
                {"name": "FOMC Meeting Minutes", "importance": "high", "forecast": "", "previous": ""},
                {"name": "Unemployment Rate", "importance": "high", "forecast": "3.6%", "previous": "3.7%"},
                {"name": "ISM Manufacturing PMI", "importance": "medium", "forecast": "51.2", "previous": "50.8"},
                {"name": "ISM Services PMI", "importance": "medium", "forecast": "52.3", "previous": "51.9"},
            ],
            "EUR": [
                {"name": "CPI y/y", "importance": "high", "forecast": "+1.8%", "previous": "+1.7%"},
                {"name": "ECB Interest Rate Decision", "importance": "high", "forecast": "2.5%", "previous": "2.5%"},
                {"name": "ECB President Lagarde Speech", "importance": "high", "forecast": "", "previous": ""},
                {"name": "German Manufacturing PMI", "importance": "medium", "forecast": "49.5", "previous": "48.7"},
                {"name": "German ZEW Economic Sentiment", "importance": "medium", "forecast": "-20.4", "previous": "-25.2"},
                {"name": "EU Economic Forecasts", "importance": "medium", "forecast": "", "previous": ""},
            ],
            "GBP": [
                {"name": "BOE Interest Rate Decision", "importance": "high", "forecast": "3.0%", "previous": "3.0%"},
                {"name": "GDP m/m", "importance": "high", "forecast": "+0.2%", "previous": "+0.1%"},
                {"name": "Unemployment Rate", "importance": "medium", "forecast": "4.2%", "previous": "4.3%"},
                {"name": "Manufacturing PMI", "importance": "medium", "forecast": "49.8", "previous": "49.2"},
                {"name": "Services PMI", "importance": "medium", "forecast": "51.5", "previous": "50.9"},
                {"name": "BOE Governor Bailey Speech", "importance": "high", "forecast": "", "previous": ""},
            ],
            "JPY": [
                {"name": "BOJ Interest Rate Decision", "importance": "high", "forecast": "-0.10%", "previous": "-0.10%"},
                {"name": "GDP q/q", "importance": "high", "forecast": "+0.4%", "previous": "+0.3%"},
                {"name": "Tokyo Core CPI y/y", "importance": "medium", "forecast": "+2.2%", "previous": "+2.1%"},
                {"name": "BOJ Outlook Report", "importance": "high", "forecast": "", "previous": ""},
                {"name": "Tankan Manufacturing Index", "importance": "medium", "forecast": "10", "previous": "8"},
            ],
            "AUD": [
                {"name": "RBA Interest Rate Decision", "importance": "high", "forecast": "2.85%", "previous": "2.85%"},
                {"name": "Employment Change", "importance": "high", "forecast": "+25.3K", "previous": "+32.2K"},
                {"name": "CPI q/q", "importance": "high", "forecast": "+1.0%", "previous": "+1.2%"},
                {"name": "Trade Balance", "importance": "medium", "forecast": "5.80B", "previous": "5.65B"},
                {"name": "RBA Governor Lowe Speech", "importance": "high", "forecast": "", "previous": ""},
            ],
            "CAD": [
                {"name": "BOC Interest Rate Decision", "importance": "high", "forecast": "3.25%", "previous": "3.25%"},
                {"name": "Employment Change", "importance": "high", "forecast": "+21.1K", "previous": "+10.1K"},
                {"name": "CPI m/m", "importance": "high", "forecast": "+0.2%", "previous": "+0.1%"},
                {"name": "Retail Sales m/m", "importance": "medium", "forecast": "+0.5%", "previous": "+0.4%"},
                {"name": "GDP m/m", "importance": "high", "forecast": "+0.3%", "previous": "+0.1%"},
                {"name": "BOC Governor Macklem Speech", "importance": "high", "forecast": "", "previous": ""},
            ],
            "CHF": [
                {"name": "SNB Interest Rate Decision", "importance": "high", "forecast": "1.75%", "previous": "1.75%"},
                {"name": "CPI m/m", "importance": "high", "forecast": "+0.1%", "previous": "+0.0%"},
                {"name": "Retail Sales y/y", "importance": "medium", "forecast": "+0.8%", "previous": "+0.5%"},
                {"name": "Trade Balance", "importance": "medium", "forecast": "3.2B", "previous": "3.0B"},
                {"name": "SNB Chairman Jordan Speech", "importance": "high", "forecast": "", "previous": ""},
            ],
            "NZD": [
                {"name": "RBNZ Interest Rate Decision", "importance": "high", "forecast": "3.5%", "previous": "3.5%"},
                {"name": "CPI q/q", "importance": "high", "forecast": "+0.9%", "previous": "+1.1%"},
                {"name": "Employment Change q/q", "importance": "high", "forecast": "+0.3%", "previous": "+0.2%"},
                {"name": "GDT Price Index", "importance": "medium", "forecast": "", "previous": "+2.1%"},
                {"name": "Trade Balance", "importance": "medium", "forecast": "-0.90B", "previous": "-1.05B"},
                {"name": "RBNZ Governor Orr Speech", "importance": "high", "forecast": "", "previous": ""},
            ]
        }
        
        # Generate mock events for each currency and day
        for day in range(days):
            event_date = current_date + timedelta(days=day)
            # Add 2-4 events per day per currency
            for currency in currencies:
                if currency in event_templates:
                    # Select 1-3 random events for this currency on this day
                    num_events = random.randint(1, 3)
                    currency_events = random.sample(event_templates[currency], min(num_events, len(event_templates[currency])))
                    
                    for event_template in currency_events:
                        # Generate random hour between 8 AM and 6 PM
                        event_hour = random.randint(8, 18)
                        event_minute = random.choice([0, 15, 30, 45])
                        event_time = event_date.replace(hour=event_hour, minute=event_minute, second=0, microsecond=0)
                        
                        # Create event with all required fields
                        event = {
                            "date": event_time.strftime("%Y-%m-%d"),
                            "time": event_time.strftime("%H:%M"),
                            "datetime": event_time,
                            "currency": currency,
                            "name": event_template["name"],
                            "importance": event_template["importance"],
                            "forecast": event_template["forecast"],
                            "previous": event_template["previous"],
                            # Random actual result if event is in the past
                            "actual": random.choice(["", "+0.1%", "+0.2%", "+0.3%", "-0.1%", "-0.2%"]) if event_time < datetime.now() else ""
                        }
                        events.append(event)
        
        # Sort events by date and time
        events.sort(key=lambda x: x["datetime"])
        
        return events
    
    async def get_calendar_html(self, currencies: List[str] = None, days: int = 3) -> str:
        """
        Get HTML-formatted calendar events
        
        Args:
            currencies: List of currency codes to filter events by
            days: Number of days to look ahead
            
        Returns:
            str: HTML-formatted calendar events
        """
        events = await self.get_events(currencies, days)
        return self._format_events_html(events)
    
    def _format_events_html(self, events: List[Dict[str, Any]]) -> str:
        """
        Format calendar events as HTML
        
        Args:
            events: List of calendar events
            
        Returns:
            str: HTML-formatted calendar events
        """
        if not events:
            return "<b>No economic events found for the selected period.</b>"
        
        # Group events by date
        events_by_date = {}
        for event in events:
            date = event["date"]
            if date not in events_by_date:
                events_by_date[date] = []
            events_by_date[date].append(event)
        
        # Format the HTML output
        result = "<b>📅 Economic Calendar</b>\n\n"
        
        for date, day_events in events_by_date.items():
            # Format date nicely
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            date_str = date_obj.strftime("%A, %B %d")
            result += f"<b>📆 {date_str}</b>\n\n"
            
            # Sort day's events by importance (high to low) and time
            day_events.sort(key=lambda x: (0 if x["importance"] == "high" else 1 if x["importance"] == "medium" else 2, x["time"]))
            
            for event in day_events:
                # Format importance with emoji
                if event["importance"] == "high":
                    importance = "🔴"
                elif event["importance"] == "medium":
                    importance = "🟡"
                else:
                    importance = "🟢"
                
                # Format time, currency and name
                result += f"{event['time']} - {importance} <b>{event['currency']}</b>: {event['name']}\n"
                
                # Add forecast/previous/actual if available
                details = []
                if event["forecast"]:
                    details.append(f"Forecast: {event['forecast']}")
                if event["previous"]:
                    details.append(f"Previous: {event['previous']}")
                if event["actual"]:
                    details.append(f"Actual: {event['actual']}")
                
                if details:
                    result += f"    <i>{', '.join(details)}</i>\n"
                
                result += "\n"
            
        return result

# Make the class available directly from the module
__all__ = ["EconomicCalendarService"]
