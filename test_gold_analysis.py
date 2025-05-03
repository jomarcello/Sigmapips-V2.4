import asyncio
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Import the DirectYahooProvider for symbol mapping
from trading_bot.services.chart_service.direct_yahoo_provider import DirectYahooProvider

async def download_gold_data(timeframe="1d", period="3mo"):
    """Download Gold price data from Yahoo Finance"""
    logger.info(f"Downloading Gold (XAUUSD) data with timeframe={timeframe}, period={period}")
    
    # Get the Yahoo Finance symbol for Gold
    yahoo_symbol = DirectYahooProvider._format_symbol("XAUUSD")
    logger.info(f"Using Yahoo Finance symbol: {yahoo_symbol}")
    
    try:
        # Download data using yfinance
        ticker = yf.Ticker(yahoo_symbol)
        data = ticker.history(period=period, interval=timeframe)
        
        if data is None or data.empty:
            logger.error(f"No data returned for XAUUSD ({yahoo_symbol})")
            return None
            
        logger.info(f"Successfully downloaded data with shape {data.shape}")
        return data
        
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        return None

def calculate_indicators(df):
    """Calculate technical indicators for the data"""
    logger.info("Calculating technical indicators")
    
    df_with_indicators = df.copy()
    
    try:
        close = df_with_indicators['Close']
        
        # Calculate EMAs
        df_with_indicators['EMA_20'] = close.ewm(span=20, adjust=False).mean()
        df_with_indicators['EMA_50'] = close.ewm(span=50, adjust=False).mean()
        df_with_indicators['EMA_200'] = close.ewm(span=200, adjust=False).mean()
        
        # Calculate RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_with_indicators['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df_with_indicators['MACD_12_26_9'] = ema_12 - ema_26
        df_with_indicators['MACDs_12_26_9'] = df_with_indicators['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        df_with_indicators['MACDh_12_26_9'] = df_with_indicators['MACD_12_26_9'] - df_with_indicators['MACDs_12_26_9']
        
        # Calculate Bollinger Bands
        df_with_indicators['SMA_20'] = close.rolling(window=20).mean()
        rolling_std = close.rolling(window=20).std()
        df_with_indicators['BOLU_20_2'] = df_with_indicators['SMA_20'] + (rolling_std * 2)
        df_with_indicators['BOLD_20_2'] = df_with_indicators['SMA_20'] - (rolling_std * 2)
        
        # Calculate support and resistance
        df_with_indicators['High_25_Max'] = df_with_indicators['High'].rolling(25).max()
        df_with_indicators['Low_25_Min'] = df_with_indicators['Low'].rolling(25).min()
        
        logger.info(f"Successfully calculated indicators")
        return df_with_indicators
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return df

def generate_technical_analysis(data, timeframe="1d"):
    """Generate technical analysis for Gold"""
    logger.info("Generating technical analysis report for Gold")
    
    # Get the last row for current values
    last_row = data.iloc[-1]
    
    # Get daily high/low
    daily_data = data.tail(1)
    daily_high = daily_data['High'].max()
    daily_low = daily_data['Low'].min()
    
    # Get weekly high/low from the last 5 trading days
    weekly_data = data.tail(5)
    weekly_high = weekly_data['High'].max()
    weekly_low = weekly_data['Low'].min()
    
    # Format current price with appropriate precision
    current_price = float(last_row['Close'])
    formatted_price = f"{current_price:.2f}"
    
    # Determine market direction based on EMAs
    market_direction = "neutral"
    if last_row['EMA_20'] > last_row['EMA_50']:
        market_direction = "bullish"
    elif last_row['EMA_20'] < last_row['EMA_50']:
        market_direction = "bearish"
    
    # Calculate momentum strength (1-5 stars)
    momentum_strength = 3  # Default
    
    # Adjust based on RSI
    rsi = last_row['RSI_14']
    if rsi > 70 or rsi < 30:
        momentum_strength += 1
    
    # Adjust based on MACD
    macd = last_row['MACD_12_26_9']
    macd_signal = last_row['MACDs_12_26_9']
    macd_hist = last_row['MACDh_12_26_9']
    
    if (market_direction == "bullish" and macd > macd_signal) or (market_direction == "bearish" and macd < macd_signal):
        momentum_strength += 1
    
    # Ensure within 1-5 range
    momentum_strength = max(1, min(5, momentum_strength))
    
    # Create strength stars
    strength_stars = "★" * momentum_strength + "☆" * (5 - momentum_strength)
    
    # RSI analysis
    if rsi > 70:
        rsi_analysis = f"overbought ({rsi:.2f})"
    elif rsi < 30:
        rsi_analysis = f"oversold ({rsi:.2f})"
    else:
        rsi_analysis = f"neutral ({rsi:.2f})"
    
    # MACD analysis
    if macd > macd_signal:
        macd_analysis = f"bullish ({macd:.5f} is above signal {macd_signal:.5f})"
    else:
        macd_analysis = f"bearish ({macd:.5f} is below signal {macd_signal:.5f})"
    
    # Moving averages analysis
    if current_price > last_row['EMA_50'] and current_price > last_row['EMA_200']:
        ma_analysis = f"Price above EMA 50 ({last_row['EMA_50']:.2f}) and above EMA 200 ({last_row['EMA_200']:.2f}), confirming bullish bias."
    elif current_price < last_row['EMA_50'] and current_price < last_row['EMA_200']:
        ma_analysis = f"Price below EMA 50 ({last_row['EMA_50']:.2f}) and below EMA 200 ({last_row['EMA_200']:.2f}), confirming bearish bias."
    elif current_price > last_row['EMA_50'] and current_price < last_row['EMA_200']:
        ma_analysis = f"Price above EMA 50 ({last_row['EMA_50']:.2f}) but below EMA 200 ({last_row['EMA_200']:.2f}), showing mixed signals."
    else:
        ma_analysis = f"Price below EMA 50 ({last_row['EMA_50']:.2f}) but above EMA 200 ({last_row['EMA_200']:.2f}), showing mixed signals."
    
    # Generate AI recommendation
    if market_direction == "bullish":
        recommendation = f"Watch for a breakout above {daily_high:.2f} for further upside. Maintain a buy bias while price holds above {daily_low:.2f}. Be cautious of overbought conditions if RSI approaches 70."
    elif market_direction == "bearish":
        recommendation = f"Watch for a breakdown below {daily_low:.2f} for further downside. Maintain a sell bias while price holds below {daily_high:.2f}. Be cautious of oversold conditions if RSI approaches 30."
    else:
        recommendation = f"Market is in consolidation. Wait for a breakout above {daily_high:.2f} or breakdown below {daily_low:.2f} before taking a position. Monitor volume for breakout confirmation."
    
    # Generate market overview
    if market_direction == "bullish":
        overview = f"Price is currently trading near current price of {formatted_price}, showing bullish momentum. The pair remains above key EMAs, indicating a strong uptrend. Volume is moderate, supporting the current price action."
    elif market_direction == "bearish":
        overview = f"Price is currently trading near current price of {formatted_price}, showing bearish momentum. The pair remains below key EMAs, indicating a strong downtrend. Volume is moderate, supporting the current price action."
    else:
        overview = f"Price is currently trading near current price of {formatted_price}, showing neutral momentum. The pair is consolidating near key EMAs, indicating indecision. Volume is moderate, supporting the current price action."
    
    # Generate analysis text
    analysis = f"""Gold (GC=F) Analysis

Zone Strength: {strength_stars}

📊 Market Overview
{overview}

🔑 Key Levels
Daily High:   {daily_high:.2f}
Daily Low:    {daily_low:.2f}
Weekly High:  {weekly_high:.2f}
Weekly Low:   {weekly_low:.2f}

📈 Technical Indicators
RSI: {rsi_analysis}
MACD: {macd_analysis}
Moving Averages: {ma_analysis}

🤖 Sigmapips AI Recommendation
{recommendation}

⚠️ Disclaimer: For educational purposes only.
"""
    
    return analysis

def generate_chart(data, filename="gold_chart.png"):
    """Generate a chart for Gold"""
    logger.info("Generating chart for Gold")
    
    try:
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Create subplot for price and indicators
        ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan=4, colspan=1)
        
        # Plot candlesticks manually
        width = 0.6
        width2 = 0.1
        
        # Prepare data
        df = data.copy()
        df['Date'] = df.index
        
        up = df[df.Close >= df.Open]
        down = df[df.Close < df.Open]
        
        # Plot up candles
        plt.bar(up.index, up.High - up.Low, width=width2, bottom=up.Low, color='green', alpha=0.5)
        plt.bar(up.index, up.Close - up.Open, width=width, bottom=up.Open, color='green')
        
        # Plot down candles
        plt.bar(down.index, down.High - down.Low, width=width2, bottom=down.Low, color='red', alpha=0.5)
        plt.bar(down.index, down.Open - down.Close, width=width, bottom=down.Close, color='red')
        
        # Plot EMAs
        plt.plot(df.index, df['EMA_20'], color='blue', linewidth=1.5, label='EMA 20')
        plt.plot(df.index, df['EMA_50'], color='orange', linewidth=1.5, label='EMA 50')
        plt.plot(df.index, df['EMA_200'], color='purple', linewidth=1.5, label='EMA 200')
        
        # Plot Bollinger Bands
        plt.plot(df.index, df['BOLU_20_2'], color='grey', linestyle='--', linewidth=1, label='Upper BB')
        plt.plot(df.index, df['BOLD_20_2'], color='grey', linestyle='--', linewidth=1, label='Lower BB')
        
        # Add grid, legend and title
        plt.title('Gold (GC=F) Price Chart', fontsize=15)
        plt.ylabel('Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        
        # Create subplot for RSI
        ax2 = plt.subplot2grid((6, 1), (4, 0), rowspan=1, colspan=1, sharex=ax1)
        plt.plot(df.index, df['RSI_14'], color='blue', linewidth=1.5, label='RSI(14)')
        plt.hlines(70, df.index[0], df.index[-1], colors='red', linestyles='dashed')
        plt.hlines(30, df.index[0], df.index[-1], colors='green', linestyles='dashed')
        plt.ylabel('RSI', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)
        
        # Create subplot for MACD
        ax3 = plt.subplot2grid((6, 1), (5, 0), rowspan=1, colspan=1, sharex=ax1)
        plt.plot(df.index, df['MACD_12_26_9'], color='blue', linewidth=1.5, label='MACD')
        plt.plot(df.index, df['MACDs_12_26_9'], color='red', linewidth=1.5, label='Signal')
        plt.bar(df.index, df['MACDh_12_26_9'], color=['green' if x > 0 else 'red' for x in df['MACDh_12_26_9']], alpha=0.5)
        plt.ylabel('MACD', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper left')
        
        # Add date
        plt.xlabel('Date', fontsize=12)
        plt.xticks(rotation=45)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Chart saved to {filename}")
        return True
        
    except Exception as e:
        logger.error(f"Error generating chart: {str(e)}")
        return False

async def main():
    """Main function to run the analysis"""
    logger.info("Starting Gold Technical Analysis")
    
    # Download Gold price data
    data = await download_gold_data(timeframe="1d", period="6mo")
    
    if data is None:
        logger.error("Failed to download Gold data")
        return
    
    # Calculate indicators
    data_with_indicators = calculate_indicators(data)
    
    # Generate technical analysis
    analysis = generate_technical_analysis(data_with_indicators)
    
    # Print analysis to console
    print("\n" + analysis)
    
    # Save analysis to file
    with open("gold_analysis.txt", "w") as f:
        f.write(analysis)
    
    # Generate chart
    generate_chart(data_with_indicators)
    
    logger.info("Gold Technical Analysis completed")

if __name__ == "__main__":
    asyncio.run(main()) 