"""
Script to collect real-time stock price data.
"""

import yfinance as yf
import pandas as pd
import requests
import datetime
import time
import logging
import sys
import os

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.api_config import (
    ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_BASE_URL,
    TARGET_STOCKS, PRICE_UPDATE_INTERVAL
)
from database.db_connector import DatabaseConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/stock_prices.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockPriceCollector:
    """Collects real-time stock price data from various APIs."""
    
    def __init__(self):
        self.db = DatabaseConnector()
    
    def fetch_alpha_vantage_data(self, symbol):
        """
        Fetch stock price data from Alpha Vantage API.
        
        Args:
            symbol (str): Stock ticker symbol
            
        Returns:
            dict: Stock price data with OHLCV (Open, High, Low, Close, Volume)
        """
        url = ALPHA_VANTAGE_BASE_URL
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            quote = data.get('Global Quote', {})
            
            if not quote:
                logger.warning(f"No data returned from Alpha Vantage for {symbol}")
                return None
            
            price_data = {
                'symbol': symbol,
                'timestamp': datetime.datetime.now(),
                'open': float(quote.get('02. open', 0)),
                'high': float(quote.get('03. high', 0)),
                'low': float(quote.get('04. low', 0)),
                'close': float(quote.get('05. price', 0)),
                'volume': int(quote.get('06. volume', 0)),
                'source': 'Alpha Vantage'
            }
            
            return price_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {e}")
            return None
        except (ValueError, KeyError) as e:
            logger.error(f"Error parsing Alpha Vantage data for {symbol}: {e}")
            return None
    
    def fetch_yahoo_finance_data(self, symbol):
        """
        Fetch stock price data from Yahoo Finance API.
        
        Args:
            symbol (str): Stock ticker symbol
            
        Returns:
            dict: Stock price data with OHLCV
        """
        try:
            # Get the most recent data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                logger.warning(f"No data returned from Yahoo Finance for {symbol}")
                return None
            
            # Get the most recent row
            last_row = hist.iloc[-1]
            
            price_data = {
                'symbol': symbol,
                'timestamp': datetime.datetime.now(),
                'open': float(last_row['Open']),
                'high': float(last_row['High']),
                'low': float(last_row['Low']),
                'close': float(last_row['Close']),
                'volume': int(last_row['Volume']),
                'source': 'Yahoo Finance'
            }
            
            return price_data
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance data for {symbol}: {e}")
            return None
    
    def run_collection(self):
        """
        Run continuous stock price collection for all target stocks.
        """
        while True:
            price_data_list = []
            
            for stock in TARGET_STOCKS:
                symbol = stock['symbol']
                
                # Try Alpha Vantage first
                price_data = self.fetch_alpha_vantage_data(symbol)
                
                # If Alpha Vantage fails, try Yahoo Finance as a backup
                if price_data is None:
                    logger.info(f"Falling back to Yahoo Finance for {symbol}")
                    price_data = self.fetch_yahoo_finance_data(symbol)
                
                if price_data:
                    price_data_list.append(price_data)
                
                # Rate limiting - be nice to the APIs
                time.sleep(1)
            
            # Store all collected price data in the database
            if price_data_list:
                logger.info(f"Storing {len(price_data_list)} price records in the database")
                self.db.insert_stock_prices(price_data_list)
            
            # Wait for the next update interval
            logger.info(f"Waiting {PRICE_UPDATE_INTERVAL} seconds until next price collection")
            time.sleep(PRICE_UPDATE_INTERVAL)
    
    def collect_latest_prices(self):
        """
        Collect the latest prices once (not continuous).
        Returns the collected price data.
        """
        price_data_list = []
        
        for stock in TARGET_STOCKS:
            symbol = stock['symbol']
            
            # Try both sources and use whichever succeeds
            price_data = self.fetch_yahoo_finance_data(symbol)
            if price_data is None:
                price_data = self.fetch_alpha_vantage_data(symbol)
            
            if price_data:
                price_data_list.append(price_data)
            
            time.sleep(0.5)  # Rate limiting
        
        # Store in the database
        if price_data_list:
            self.db.insert_stock_prices(price_data_list)
        
        return price_data_list

if __name__ == "__main__":
    collector = StockPriceCollector()
    collector.run_collection()