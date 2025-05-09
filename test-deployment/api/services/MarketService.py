from api.services.ChartService import ChartService

class MarketService:
    def __init__(self):
        # Dit zal een circulaire import veroorzaken
        self.chart_service = ChartService()
    
    def get_market_data(self, symbol):
        # Fetch market data from external source
        return {
            "symbol": symbol,
            "price_data": [100.0, 101.2, 102.5, 101.8, 103.2, 105.1, 104.8],
            "volume": [1000, 1200, 1150, 980, 1050, 1300, 1100],
            "last_ema": 103.5
        }
    
    def get_market_summary(self, symbol):
        # Use chart service to get chart data
        chart_data = self.chart_service.get_chart_data(symbol, "1d")
        
        # Create a summary
        return {
            "symbol": symbol,
            "current_price": chart_data["data"][-1],
            "sma": chart_data["indicators"]["sma"],
            "ema": chart_data["indicators"]["ema"],
            "trend": "up" if chart_data["data"][-1] > chart_data["data"][0] else "down"
        } 