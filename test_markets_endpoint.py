import http.client
import json
import sys
import os

# Show full output for large data
pd_width = 320
pd_max_rows = 100
pd_max_cols = 20

# Set API credentials
API_KEY = os.environ.get("RAPIDAPI_KEY", "c348ca56d4msh3adcf951c16e7cf1f1fad9jsn81056145fe21")
API_HOST = "yahoo-finance15.p.rapidapi.com"

def test_markets_endpoint():
    """Test the markets endpoint using the exact approach from the example"""
    print("Testing markets endpoint with http.client...")
    
    try:
        # Connect to the API
        conn = http.client.HTTPSConnection(API_HOST)
        
        # Define headers with API key
        headers = {
            'X-RapidAPI-Key': API_KEY,
            'X-RapidAPI-Host': API_HOST
        }
        
        # Make GET request to the tickers endpoint for stocks
        print("Making request to /api/v2/markets/tickers?page=1&type=STOCKS")
        conn.request("GET", "/api/v2/markets/tickers?page=1&type=STOCKS", headers=headers)
        
        # Get response
        res = conn.getresponse()
        data = res.read()
        
        # Check the status code
        print(f"Response status: {res.status} {res.reason}")
        
        # If successful, display some of the result
        if res.status == 200:
            try:
                json_data = json.loads(data.decode('utf-8'))
                print(f"Successfully retrieved data. Response size: {len(data)} bytes")
                
                # Print the first part of the data to see the structure
                if 'data' in json_data:
                    print(f"Number of items: {len(json_data['data'])}")
                    print("\nFirst 5 items:")
                    for i, item in enumerate(json_data['data'][:5]):
                        print(f"{i+1}. {item}")
                else:
                    print("Data structure:", json.dumps(json_data, indent=2)[:1000] + "...")
            
            except json.JSONDecodeError:
                print("Failed to decode JSON response")
                print("Raw response:", data.decode('utf-8')[:1000])
        else:
            print("Error response:", data.decode('utf-8')[:1000])
    
    except Exception as e:
        print(f"Error: {str(e)}")

def test_historical_endpoint():
    """Test historical data endpoint for a single stock"""
    print("\nTesting historical data endpoint for AAPL...")
    
    try:
        conn = http.client.HTTPSConnection(API_HOST)
        
        headers = {
            'X-RapidAPI-Key': API_KEY,
            'X-RapidAPI-Host': API_HOST
        }
        
        # Test API for historical prices
        endpoint = "/api/yahoo/mo/module/AAPL?module=historical-prices"
        print(f"Making request to {endpoint}")
        conn.request("GET", endpoint, headers=headers)
        
        res = conn.getresponse()
        data = res.read()
        
        print(f"Response status: {res.status} {res.reason}")
        
        if res.status == 200:
            try:
                json_data = json.loads(data.decode('utf-8'))
                print(f"Successfully retrieved data. Response size: {len(data)} bytes")
                
                # Navigate the structure to find historical prices
                if 'dispatcher' in json_data and 'stores' in json_data['dispatcher']:
                    stores = json_data['dispatcher']['stores']
                    if 'HistoricalPriceStore' in stores and 'prices' in stores['HistoricalPriceStore']:
                        prices = stores['HistoricalPriceStore']['prices']
                        print(f"Found {len(prices)} historical price records")
                        if prices:
                            print("\nFirst 3 price records:")
                            for i, price in enumerate(prices[:3]):
                                print(f"{i+1}. {price}")
                    else:
                        print("Historical price data not found in response structure")
                else:
                    print("Data structure doesn't match expected format")
                    print("Response structure:", json.dumps(json_data, indent=2)[:1000] + "...")
            
            except json.JSONDecodeError:
                print("Failed to decode JSON response")
                print("Raw response:", data.decode('utf-8')[:1000])
        else:
            print("Error response:", data.decode('utf-8')[:1000])
    
    except Exception as e:
        print(f"Error: {str(e)}")

def test_chart_endpoint():
    """Test chart data endpoint for a single stock"""
    print("\nTesting chart endpoint for AAPL...")
    
    try:
        conn = http.client.HTTPSConnection(API_HOST)
        
        headers = {
            'X-RapidAPI-Key': API_KEY,
            'X-RapidAPI-Host': API_HOST
        }
        
        # Test API for chart data
        endpoint = "/api/yahoo/v1/chart/AAPL?range=5d&interval=1h&events=div%2Csplit"
        print(f"Making request to {endpoint}")
        conn.request("GET", endpoint, headers=headers)
        
        res = conn.getresponse()
        data = res.read()
        
        print(f"Response status: {res.status} {res.reason}")
        
        if res.status == 200:
            try:
                json_data = json.loads(data.decode('utf-8'))
                print(f"Successfully retrieved data. Response size: {len(data)} bytes")
                
                # Check if we have chart data
                if 'chart' in json_data and 'result' in json_data['chart'] and json_data['chart']['result']:
                    chart_data = json_data['chart']['result'][0]
                    timestamp_count = len(chart_data.get('timestamp', []))
                    quote_data = chart_data.get('indicators', {}).get('quote', [{}])[0]
                    
                    print(f"Found {timestamp_count} data points")
                    
                    if timestamp_count > 0:
                        print("\nSample data (first 3 points):")
                        for i in range(min(3, timestamp_count)):
                            ts = chart_data['timestamp'][i]
                            o = quote_data.get('open', [])[i] if 'open' in quote_data and i < len(quote_data['open']) else None
                            h = quote_data.get('high', [])[i] if 'high' in quote_data and i < len(quote_data['high']) else None
                            l = quote_data.get('low', [])[i] if 'low' in quote_data and i < len(quote_data['low']) else None
                            c = quote_data.get('close', [])[i] if 'close' in quote_data and i < len(quote_data['close']) else None
                            v = quote_data.get('volume', [])[i] if 'volume' in quote_data and i < len(quote_data['volume']) else None
                            
                            print(f"{i+1}. Time: {ts} - OHLCV: {o}, {h}, {l}, {c}, {v}")
                else:
                    print("Chart data not found in response structure")
                    print("Response structure:", json.dumps(json_data, indent=2)[:1000] + "...")
            
            except json.JSONDecodeError:
                print("Failed to decode JSON response")
                print("Raw response:", data.decode('utf-8')[:1000])
        else:
            print("Error response:", data.decode('utf-8')[:1000])
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_markets_endpoint()
    test_historical_endpoint()
    test_chart_endpoint() 