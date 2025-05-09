#!/usr/bin/env python3
"""
Main application entry point for Nixpacks
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("app")

async def run_test():
    """Simple test function"""
    logger.info("Running test")
    return True

if __name__ == "__main__":
    logger.info("Starting application")
    asyncio.run(run_test())
    logger.info("Application completed")
