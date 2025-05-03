import asyncio
import logging
import sys
import os

# Ensure the trading_bot package is in the Python path
# Adjust the path depth if your script is located elsewhere relative to the trading_bot directory
project_root = os.path.dirname(os.path.abspath(__file__))
# If the script is in the root, and trading_bot is a subdirectory
sys.path.insert(0, project_root)
# If the script is in a 'tests' subdirectory, use os.path.join(project_root, '..')

try:
    from trading_bot.services.chart_service.chart import ChartService
except ImportError as e:
    print(f"Error importing ChartService: {e}")
    print("Please ensure the script is run from the project root directory or adjust the sys.path.")
    sys.exit(1)

# Configure logging to see provider details and potential errors
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Output logs to console
    ]
)
logger = logging.getLogger(__name__)

async def run_test():
    """Runs the Yahoo Finance integration test."""
    logger.info("Starting Yahoo Finance integration test...")
    chart_service = ChartService()

    # Although initialize might be called implicitly, call it explicitly for clarity
    logger.info("Initializing ChartService...")
    try:
        # Note: The initialize method in the provided chart.py always returns True
        await chart_service.initialize()
        logger.info("ChartService initialized.")
    except Exception as e:
        logger.error(f"Error initializing ChartService: {e}", exc_info=True)
        return # Cannot proceed without initialization

    # --- Instruments to test ---
    # Only test EURUSD as requested
    test_instruments = {
        "Forex": ["EURUSD"],
        # "Indices": ["US30", "US500", "DE40", "UK100", "JP225"],
        # "Commodities": ["XAUUSD", "XAGUSD", "XTIUSD", "XBRUSD"] # WTI and Brent Crude
        # "Crypto (Fallback Check)": ["BTCUSD", "ETHUSD"] # Optionally check if non-Yahoo works
    }

    timeframe = "1h" # Or "4h", "1d"

    all_passed = True

    for category, instruments in test_instruments.items():
        logger.info(f"\n--- Testing Category: {category} ---")
        for instrument in instruments:
            logger.info(f"Requesting Technical Analysis for: {instrument} ({timeframe})")
            try:
                analysis_text = await chart_service.get_technical_analysis(instrument, timeframe)

                if analysis_text and f"{instrument} - {timeframe}" in analysis_text and "not available" not in analysis_text:
                    logger.info(f"Successfully received analysis for {instrument}")
                    # Basic check on content - look for key sections
                    if "Market Overview" in analysis_text and "Key Levels" in analysis_text and "Technical Indicators" in analysis_text:
                         logger.info(f"[{instrument}] Analysis seems valid (contains expected sections). PASS")
                         # print(f"\n--- Analysis for {instrument} ---\n{analysis_text[:500]}...\n") # Uncomment to print part of the analysis
                    else:
                         logger.warning(f"[{instrument}] Analysis received, but might be incomplete or default. CHECK")
                         print(f"\n--- Analysis for {instrument} ---\n{analysis_text}\n") # Print full analysis for checking
                         all_passed = False # Mark as needing check

                elif analysis_text and "not available" in analysis_text:
                    logger.error(f"[{instrument}] Failed to get analysis (received 'not available' message). FAIL")
                    all_passed = False
                else:
                    logger.error(f"[{instrument}] Failed to get analysis (empty or unexpected response). FAIL")
                    all_passed = False

            except Exception as e:
                logger.error(f"[{instrument}] Exception during get_technical_analysis: {e}", exc_info=True)
                all_passed = False

            # --- Test get_chart ---
            logger.info(f"Requesting Chart Image for: {instrument} ({timeframe})")
            try:
                chart_bytes = await chart_service.get_chart(instrument, timeframe, fullscreen=True)

                if chart_bytes and isinstance(chart_bytes, bytes) and len(chart_bytes) > 1000: # Check if we got substantial bytes (more than a tiny empty image)
                    logger.info(f"[{instrument}] Successfully received chart image ({len(chart_bytes)} bytes). PASS")
                    # Save the chart for manual verification
                    try:
                        output_filename = f"test_chart_{instrument.lower()}.png"
                        with open(output_filename, "wb") as f:
                            f.write(chart_bytes)
                        logger.info(f"[{instrument}] Chart image saved to {output_filename}")
                    except Exception as save_e:
                        logger.error(f"[{instrument}] Failed to save chart image: {save_e}")
                        all_passed = False # Mark as fail if saving fails
                elif chart_bytes:
                    logger.warning(f"[{instrument}] Received chart image, but it seems very small ({len(chart_bytes)} bytes). May be fallback or error image. CHECK")
                    all_passed = False # Mark as needing check if image is too small
                else:
                    logger.error(f"[{instrument}] Failed to get chart image (received None or empty bytes). FAIL")
                    all_passed = False

            except Exception as e:
                logger.error(f"[{instrument}] Exception during get_chart: {e}", exc_info=True)
                all_passed = False

            # Add a small delay between requests to potentially avoid rate limiting
            await asyncio.sleep(1)

    logger.info("\n--- Test Summary ---")
    if all_passed:
        logger.info("All tested instruments returned an analysis. Please review logs for warnings ('CHECK').")
    else:
        logger.error("Some instruments failed to return a valid analysis. Please check the logs above for details.")

    # Cleanup resources if the service has a cleanup method
    if hasattr(chart_service, 'cleanup'):
        await chart_service.cleanup()

if __name__ == "__main__":
    asyncio.run(run_test()) 