import asyncio
import logging
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Test commodity instruments with direct Yahoo Finance futures symbols
COMMODITIES = {
    "XAUUSD": "GC=F",  # Gold futures
    "XTIUSD": "CL=F",  # WTI Crude Oil futures
    "XAGUSD": "SI=F",  # Silver futures
    "XBRUSD": "BZ=F"   # Brent Oil futures
}

async def test_download_data():
    """Test downloading data for commodity futures"""
    logger.info("Testing direct download of commodity futures data")
    
    for instrument, symbol in COMMODITIES.items():
        logger.info(f"Testing download for {instrument} using {symbol}")
        
        try:
            # Download data directly with yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1mo", interval="1d")
            
            if data is not None and not data.empty:
                logger.info(f"✅ Successfully downloaded data for {instrument} ({symbol}) with shape {data.shape}")
                logger.info(f"First row: {data.iloc[0]}")
                logger.info(f"Last row: {data.iloc[-1]}")
            else:
                logger.error(f"❌ No data returned for {instrument} ({symbol})")
                
        except Exception as e:
            logger.error(f"❌ Error downloading data for {instrument} ({symbol}): {str(e)}")
            
        # Add a small delay between requests
        await asyncio.sleep(2)

async def test_generate_chart():
    """Test generating charts for commodity futures"""
    logger.info("\nTesting chart generation for commodity futures")
    
    for instrument, symbol in COMMODITIES.items():
        logger.info(f"Generating chart for {instrument} using {symbol}")
        
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1mo", interval="1d")
            
            if df is None or df.empty:
                logger.error(f"❌ No data returned for {instrument} ({symbol})")
                continue
                
            # Generate chart
            data = df.tail(100)
            figsize = (10, 6)
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot close price
            data['Close'].plot(ax=ax, color='blue', linewidth=1.5)
            
            # Add EMAs
            data['EMA20'] = data['Close'].ewm(span=20, adjust=False).mean()
            data['EMA50'] = data['Close'].ewm(span=50, adjust=False).mean()
            data['EMA20'].plot(ax=ax, color='orange', linewidth=1.0, label='EMA 20')
            data['EMA50'].plot(ax=ax, color='red', linewidth=1.0, label='EMA 50')
            
            # Set labels
            ax.set_title(f"{instrument} - Daily Chart", fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.autofmt_xdate()
            
            # Save chart
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            
            # Save to file
            buf.seek(0)
            chart_data = buf.getvalue()
            chart_filename = f"direct_commodity_{instrument.lower()}.png"
            
            with open(chart_filename, "wb") as f:
                f.write(chart_data)
                
            file_size = os.path.getsize(chart_filename)
            logger.info(f"✅ Chart for {instrument} generated successfully and saved as {chart_filename} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"❌ Error generating chart for {instrument} ({symbol}): {str(e)}")
            
        # Add a small delay between requests
        await asyncio.sleep(2)

async def main():
    """Main function that runs all tests"""
    logger.info("Starting direct commodity futures tests")
    
    # Test downloading data
    await test_download_data()
    
    # Test generating charts
    await test_generate_chart()
    
    logger.info("Direct commodity futures tests complete")

if __name__ == "__main__":
    asyncio.run(main()) 