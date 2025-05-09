from api.services.MarketService import MarketService

class ChartService:
    def __init__(self):
        self.market_service = MarketService()

    def get_chart_data(self, symbol, timeframe):
        # Get market data first
        market_data = self.market_service.get_market_data(symbol)
        
        # Then process it for charts
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": market_data["price_data"],
            "indicators": self._calculate_indicators(market_data)
        }
    
    def _calculate_indicators(self, market_data):
        # Calculate some indicators
        return {
            "sma": sum(market_data["price_data"][-20:]) / 20,
            "ema": market_data["price_data"][-1] * 0.2 + market_data["last_ema"] * 0.8 if "last_ema" in market_data else market_data["price_data"][-1]
        } 