from api.services.ChartService import ChartService

def main():
    print("Starten van de test applicatie...")
    
    try:
        # Dit zal een circulaire import fout veroorzaken
        chart_service = ChartService()
        
        # Haal chart data op
        chart_data = chart_service.get_chart_data("BTC/USD", "1h")
        
        print(f"Chart data voor BTC/USD: {chart_data}")
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    main() 