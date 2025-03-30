"""
Script to collect financial news from various APIs.
"""

import requests
import datetime
import time
import logging
import pandas as pd
import sys
import os

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.api_config import (
    FINNHUB_API_KEY, FINNHUB_BASE_URL,
    NEWS_API_KEY, NEWS_API_BASE_URL,
    TARGET_STOCKS, NEWS_UPDATE_INTERVAL
)
from database.db_connector import DatabaseConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/financial_news.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinancialNewsCollector:
    """Collects financial news from various API sources."""
    
    def __init__(self):
        self.db = DatabaseConnector()
        self.collected_news_ids = set()  # To avoid duplicates
        
    def fetch_finnhub_news(self, symbol):
        """
        Fetch news for a specific stock symbol from Finnhub API.
        
        Args:
            symbol (str): Stock ticker symbol
            
        Returns:
            list: List of news articles
        """
        today = datetime.datetime.now()
        one_week_ago = today - datetime.timedelta(days=7)
        
        url = f"{FINNHUB_BASE_URL}/company-news"
        params = {
            'symbol': symbol,
            'from': one_week_ago.strftime('%Y-%m-%d'),
            'to': today.strftime('%Y-%m-%d'),
            'token': FINNHUB_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            news_data = response.json()
            logger.info(f"Retrieved {len(news_data)} news items from Finnhub for {symbol}")
            
            # Format the data for our database
            formatted_news = []
            for news in news_data:
                if news['id'] not in self.collected_news_ids:
                    formatted_news.append({
                        'news_id': news['id'],
                        'source': 'Finnhub',
                        'symbol': symbol,
                        'headline': news['headline'],
                        'summary': news['summary'],
                        'url': news['url'],
                        'published_at': datetime.datetime.fromtimestamp(news['datetime']),
                        'collected_at': datetime.datetime.now()
                    })
                    self.collected_news_ids.add(news['id'])
            
            return formatted_news
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Finnhub news for {symbol}: {e}")
            return []
    
    def fetch_newsapi_news(self, company_name):
        """
        Fetch news for a specific company from NewsAPI.
        
        Args:
            company_name (str): Company name
            
        Returns:
            list: List of news articles
        """
        url = f"{NEWS_API_BASE_URL}/everything"
        
        # Current date and 2 days ago date for the query
        today = datetime.datetime.now()
        two_days_ago = today - datetime.timedelta(days=2)
        
        params = {
            'q': f'"{company_name}" AND (stock OR shares OR market OR trading)',
            'from': two_days_ago.strftime('%Y-%m-%d'),
            'to': today.strftime('%Y-%m-%d'),
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': NEWS_API_KEY
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            news_data = response.json()
            articles = news_data.get('articles', [])
            logger.info(f"Retrieved {len(articles)} news items from NewsAPI for {company_name}")
            
            # Generate a unique ID for each article
            formatted_news = []
            for article in articles:
                # Create a unique ID from URL (not perfect but workable)
                news_id = hash(article['url'])
                
                if news_id not in self.collected_news_ids:
                    published_at = datetime.datetime.strptime(
                        article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'
                    ) if article['publishedAt'] else datetime.datetime.now()
                    
                    formatted_news.append({
                        'news_id': news_id,
                        'source': f"NewsAPI-{article['source']['name']}",
                        'symbol': next((stock['symbol'] for stock in TARGET_STOCKS 
                                       if stock['name'] == company_name), None),
                        'headline': article['title'],
                        'summary': article['description'],
                        'url': article['url'],
                        'published_at': published_at,
                        'collected_at': datetime.datetime.now()
                    })
                    self.collected_news_ids.add(news_id)
            
            return formatted_news
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching NewsAPI news for {company_name}: {e}")
            return []
    
    def run_collection(self):
        """
        Run continuous news collection for all target stocks.
        """
        while True:
            all_news = []
            
            # Collect news from Finnhub for all target stocks
            for stock in TARGET_STOCKS:
                symbol = stock['symbol']
                finnhub_news = self.fetch_finnhub_news(symbol)
                all_news.extend(finnhub_news)
                
                # Rate limiting - be nice to the API
                time.sleep(1)
            
            # Collect news from NewsAPI for all target companies
            for stock in TARGET_STOCKS:
                company_name = stock['name']
                newsapi_news = self.fetch_newsapi_news(company_name)
                all_news.extend(newsapi_news)
                
                # Rate limiting - be nice to the API
                time.sleep(1)
            
            # Store all collected news in the database
            if all_news:
                logger.info(f"Storing {len(all_news)} news articles in the database")
                self.db.insert_news_articles(all_news)
            
            # Wait for the next update interval
            logger.info(f"Waiting {NEWS_UPDATE_INTERVAL} seconds until next news collection")
            time.sleep(NEWS_UPDATE_INTERVAL)
    
    def collect_latest_news(self):
        """
        Collect the latest news once (not continuous).
        Returns the collected news articles.
        """
        all_news = []
        
        # Collect from Finnhub
        for stock in TARGET_STOCKS:
            finnhub_news = self.fetch_finnhub_news(stock['symbol'])
            all_news.extend(finnhub_news)
            time.sleep(0.5)  # Rate limiting
        
        # Collect from NewsAPI
        for stock in TARGET_STOCKS:
            newsapi_news = self.fetch_newsapi_news(stock['name'])
            all_news.extend(newsapi_news)
            time.sleep(0.5)  # Rate limiting
        
        # Store in the database
        if all_news:
            self.db.insert_news_articles(all_news)
        
        return all_news

if __name__ == "__main__":
    collector = FinancialNewsCollector()
    collector.run_collection()