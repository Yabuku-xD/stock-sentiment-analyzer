"""
Database connector module for handling database operations.
"""

import os
import sys
import time
import logging
import psycopg2
import psycopg2.extras
import datetime
from psycopg2.pool import ThreadedConnectionPool

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.db_config import POSTGRES_CONFIG, TABLES, BATCH_SIZE, MAX_RETRIES, RETRY_DELAY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseConnector:
    """Database connector class for handling database operations."""
    
    def __init__(self):
        """Initialize the database connector."""
        self.config = POSTGRES_CONFIG
        self.tables = TABLES
        self.pool = None
        self._initialize_connection_pool()
    
    def _initialize_connection_pool(self):
        """Initialize the connection pool."""
        for attempt in range(MAX_RETRIES):
            try:
                self.pool = ThreadedConnectionPool(
                    minconn=1,
                    maxconn=10,
                    host=self.config['host'],
                    database=self.config['database'],
                    user=self.config['user'],
                    password=self.config['password'],
                    port=self.config['port']
                )
                logger.info("Database connection pool initialized successfully")
                return
            except psycopg2.Error as e:
                logger.error(f"Failed to initialize connection pool (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("Max retries reached, could not connect to database")
                    raise
    
    def _get_connection(self):
        """
        Get a connection from the pool.
        
        Returns:
            connection: Database connection
        """
        if self.pool is None:
            self._initialize_connection_pool()
        
        for attempt in range(MAX_RETRIES):
            try:
                return self.pool.getconn()
            except psycopg2.Error as e:
                logger.error(f"Failed to get connection (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("Max retries reached, could not get database connection")
                    raise
    
    def _return_connection(self, conn):
        """
        Return a connection to the pool.
        
        Args:
            conn: Database connection to return
        """
        if self.pool is not None:
            self.pool.putconn(conn)
    
    def execute_query(self, query, params=None, fetch=True):
        """
        Execute a SQL query.
        
        Args:
            query (str): SQL query to execute
            params (tuple/dict): Parameters for the query
            fetch (bool): Whether to fetch results
            
        Returns:
            list: Query results if fetch is True, otherwise None
        """
        conn = None
        cursor = None
        result = None
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute(query, params)
            
            if fetch:
                result = cursor.fetchall()
            else:
                conn.commit()
            
            return result
            
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._return_connection(conn)
    
    def execute_batch(self, query, param_list):
        """
        Execute a batch of SQL commands.
        
        Args:
            query (str): SQL query template
            param_list (list): List of parameter tuples/dictionaries
            
        Returns:
            int: Number of affected rows
        """
        conn = None
        cursor = None
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Process in batches
            rows_affected = 0
            for i in range(0, len(param_list), BATCH_SIZE):
                batch = param_list[i:i + BATCH_SIZE]
                psycopg2.extras.execute_batch(cursor, query, batch)
                rows_affected += len(batch)
            
            conn.commit()
            return rows_affected
            
        except psycopg2.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database batch error: {e}")
            raise
            
        finally:
            if cursor:
                cursor.close()
            if conn:
                self._return_connection(conn)
    
    # Stock Prices Operations
    
    def insert_stock_prices(self, price_data_list):
        """
        Insert stock price data into the database.
        
        Args:
            price_data_list (list): List of stock price data dictionaries
            
        Returns:
            int: Number of inserted records
        """
        if not price_data_list:
            return 0
        
        query = f"""
        INSERT INTO {self.tables['stock_prices']} 
        (symbol, timestamp, open, high, low, close, volume, source)
        VALUES (%(symbol)s, %(timestamp)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volume)s, %(source)s)
        ON CONFLICT DO NOTHING
        """
        
        try:
            return self.execute_batch(query, price_data_list)
        except Exception as e:
            logger.error(f"Error inserting stock prices: {e}")
            return 0
    
    def get_stock_prices(self, symbol=None, start_date=None, end_date=None, limit=1000):
        """
        Get stock price data from the database.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for filtering
            end_date (datetime): End date for filtering
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of stock price records
        """
        query = f"SELECT * FROM {self.tables['stock_prices']}"
        conditions = []
        params = {}
        
        if symbol:
            conditions.append("symbol = %(symbol)s")
            params['symbol'] = symbol
        
        if start_date:
            conditions.append("timestamp >= %(start_date)s")
            params['start_date'] = start_date
        
        if end_date:
            conditions.append("timestamp <= %(end_date)s")
            params['end_date'] = end_date
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT %(limit)s"
            params['limit'] = limit
        
        try:
            result = self.execute_query(query, params)
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Error getting stock prices: {e}")
            return []
    
    # News Articles Operations
    
    def insert_news_articles(self, news_articles):
        """
        Insert news articles into the database.
        
        Args:
            news_articles (list): List of news article dictionaries
            
        Returns:
            int: Number of inserted records
        """
        if not news_articles:
            return 0
        
        query = f"""
        INSERT INTO {self.tables['news_articles']} 
        (news_id, source, symbol, headline, summary, url, published_at, collected_at)
        VALUES (%(news_id)s, %(source)s, %(symbol)s, %(headline)s, %(summary)s, 
                %(url)s, %(published_at)s, %(collected_at)s)
        ON CONFLICT (news_id) DO NOTHING
        """
        
        try:
            return self.execute_batch(query, news_articles)
        except Exception as e:
            logger.error(f"Error inserting news articles: {e}")
            return 0
    
    def get_news_articles(self, symbol=None, start_date=None, end_date=None, limit=1000):
        """
        Get news articles from the database.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for filtering
            end_date (datetime): End date for filtering
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of news article records
        """
        query = f"SELECT * FROM {self.tables['news_articles']}"
        conditions = []
        params = {}
        
        if symbol:
            conditions.append("symbol = %(symbol)s")
            params['symbol'] = symbol
        
        if start_date:
            conditions.append("published_at >= %(start_date)s")
            params['start_date'] = start_date
        
        if end_date:
            conditions.append("published_at <= %(end_date)s")
            params['end_date'] = end_date
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY published_at DESC"
        
        if limit:
            query += " LIMIT %(limit)s"
            params['limit'] = limit
        
        try:
            result = self.execute_query(query, params)
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Error getting news articles: {e}")
            return []
    
    # Social Media Posts Operations
    
    def insert_social_media_posts(self, posts):
        """
        Insert social media posts into the database.
        
        Args:
            posts (list): List of social media post dictionaries
            
        Returns:
            int: Number of inserted records
        """
        if not posts:
            return 0
        
        query = f"""
        INSERT INTO {self.tables['social_media_posts']} 
        (post_id, platform, subreddit, symbol, title, content, url, 
         user_name, upvotes, comments, created_at, collected_at)
        VALUES (%(post_id)s, %(platform)s, %(subreddit)s, %(symbol)s, %(title)s, 
                %(content)s, %(url)s, %(user)s, %(upvotes)s, %(comments)s, 
                %(created_at)s, %(collected_at)s)
        ON CONFLICT (post_id) DO NOTHING
        """
        
        try:
            return self.execute_batch(query, posts)
        except Exception as e:
            logger.error(f"Error inserting social media posts: {e}")
            return 0
    
    def get_social_media_posts(self, symbol=None, platform=None, start_date=None, end_date=None, limit=1000):
        """
        Get social media posts from the database.
        
        Args:
            symbol (str): Stock ticker symbol
            platform (str): Social media platform
            start_date (datetime): Start date for filtering
            end_date (datetime): End date for filtering
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of social media post records
        """
        query = f"SELECT * FROM {self.tables['social_media_posts']}"
        conditions = []
        params = {}
        
        if symbol:
            conditions.append("symbol = %(symbol)s")
            params['symbol'] = symbol
        
        if platform:
            conditions.append("platform = %(platform)s")
            params['platform'] = platform
        
        if start_date:
            conditions.append("created_at >= %(start_date)s")
            params['start_date'] = start_date
        
        if end_date:
            conditions.append("created_at <= %(end_date)s")
            params['end_date'] = end_date
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC"
        
        if limit:
            query += " LIMIT %(limit)s"
            params['limit'] = limit
        
        try:
            result = self.execute_query(query, params)
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Error getting social media posts: {e}")
            return []
    
    # Sentiment Scores Operations
    
    def insert_sentiment_scores(self, sentiment_scores):
        """
        Insert sentiment scores into the database.
        
        Args:
            sentiment_scores (list): List of sentiment score dictionaries
            
        Returns:
            int: Number of inserted records
        """
        if not sentiment_scores:
            return 0
        
        query = f"""
        INSERT INTO {self.tables['sentiment_scores']} 
        (reference_id, symbol, source, timestamp, content_type, 
         compound_score, positive_score, negative_score, neutral_score, 
         sentiment_label, analyzed_at)
        VALUES (%(news_id)s, %(symbol)s, %(source)s, %(timestamp)s, %(content_type)s, 
                %(compound_score)s, %(positive_score)s, %(negative_score)s, %(neutral_score)s, 
                %(sentiment_label)s, %(analyzed_at)s)
        ON CONFLICT DO NOTHING
        """
        
        # Some records may use post_id instead of news_id
        for score in sentiment_scores:
            if 'news_id' not in score and 'post_id' in score:
                score['news_id'] = score['post_id']
        
        try:
            return self.execute_batch(query, sentiment_scores)
        except Exception as e:
            logger.error(f"Error inserting sentiment scores: {e}")
            return 0
    
    def get_sentiment_scores(self, symbol=None, content_type=None, start_date=None, end_date=None, limit=1000):
        """
        Get sentiment scores from the database.
        
        Args:
            symbol (str): Stock ticker symbol
            content_type (str): Content type ('news' or 'social')
            start_date (datetime): Start date for filtering
            end_date (datetime): End date for filtering
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of sentiment score records
        """
        query = f"SELECT * FROM {self.tables['sentiment_scores']}"
        conditions = []
        params = {}
        
        if symbol:
            conditions.append("symbol = %(symbol)s")
            params['symbol'] = symbol
        
        if content_type:
            conditions.append("content_type = %(content_type)s")
            params['content_type'] = content_type
        
        if start_date:
            conditions.append("timestamp >= %(start_date)s")
            params['start_date'] = start_date
        
        if end_date:
            conditions.append("timestamp <= %(end_date)s")
            params['end_date'] = end_date
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT %(limit)s"
            params['limit'] = limit
        
        try:
            result = self.execute_query(query, params)
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Error getting sentiment scores: {e}")
            return []
    
    # Entities Operations
    
    def insert_entities(self, entities):
        """
        Insert extracted entities into the database.
        
        Args:
            entities (list): List of entity dictionaries
            
        Returns:
            int: Number of inserted records
        """
        if not entities:
            return 0
        
        query = f"""
        INSERT INTO {self.tables['entities']} 
        (reference_id, source, entity_type, entity_value, entity_text, timestamp, extracted_at)
        VALUES (%(reference_id)s, %(source)s, %(entity_type)s, %(entity_value)s, 
                %(entity_text)s, %(timestamp)s, %(extracted_at)s)
        ON CONFLICT DO NOTHING
        """
        
        try:
            return self.execute_batch(query, entities)
        except Exception as e:
            logger.error(f"Error inserting entities: {e}")
            return 0
    
    def get_entities(self, entity_type=None, entity_value=None, start_date=None, end_date=None, limit=1000):
        """
        Get entities from the database.
        
        Args:
            entity_type (str): Type of entity
            entity_value (str): Value of entity
            start_date (datetime): Start date for filtering
            end_date (datetime): End date for filtering
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of entity records
        """
        query = f"SELECT * FROM {self.tables['entities']}"
        conditions = []
        params = {}
        
        if entity_type:
            conditions.append("entity_type = %(entity_type)s")
            params['entity_type'] = entity_type
        
        if entity_value:
            conditions.append("entity_value = %(entity_value)s")
            params['entity_value'] = entity_value
        
        if start_date:
            conditions.append("timestamp >= %(start_date)s")
            params['start_date'] = start_date
        
        if end_date:
            conditions.append("timestamp <= %(end_date)s")
            params['end_date'] = end_date
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT %(limit)s"
            params['limit'] = limit
        
        try:
            result = self.execute_query(query, params)
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Error getting entities: {e}")
            return []
    
    # Correlations Operations
    
    def insert_correlation(self, correlation_data):
        """
        Insert correlation data into the database.
        
        Args:
            correlation_data (dict): Correlation data dictionary
            
        Returns:
            bool: Success status
        """
        query = f"""
        INSERT INTO {self.tables['correlations']}
        (symbol, start_date, end_date, window_size, price_change, 
        sentiment_change, correlation_value, data_points, calculated_at)
        VALUES
        (%(symbol)s, %(start_date)s, %(end_date)s, %(window_size)s, 
        %(price_change)s, %(sentiment_change)s, %(correlation_value)s, 
        %(data_points)s, %(calculated_at)s)
        """
        
        try:
            self.execute_query(query, correlation_data, fetch=False)
            return True
        except Exception as e:
            logger.error(f"Error inserting correlation: {e}")
            return False
    
    def get_correlations(self, symbol=None, window_size=None, start_date=None, end_date=None, limit=100):
        """
        Get correlations from the database.
        
        Args:
            symbol (str): Stock ticker symbol
            window_size (int): Window size in minutes
            start_date (datetime): Start date for filtering
            end_date (datetime): End date for filtering
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of correlation records
        """
        query = f"SELECT * FROM {self.tables['correlations']}"
        conditions = []
        params = {}
        
        if symbol:
            conditions.append("symbol = %(symbol)s")
            params['symbol'] = symbol
        
        if window_size:
            conditions.append("window_size = %(window_size)s")
            params['window_size'] = window_size
        
        if start_date:
            conditions.append("end_date >= %(start_date)s")
            params['start_date'] = start_date
        
        if end_date:
            conditions.append("start_date <= %(end_date)s")
            params['end_date'] = end_date
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY calculated_at DESC"
        
        if limit:
            query += " LIMIT %(limit)s"
            params['limit'] = limit
        
        try:
            result = self.execute_query(query, params)
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Error getting correlations: {e}")
            return []
    
    # Summary Statistics Operations
    
    def insert_summary_statistics(self, stats_data):
        """
        Insert summary statistics into the database.
        
        Args:
            stats_data (dict): Summary statistics dictionary
            
        Returns:
            bool: Success status
        """
        query = f"""
        INSERT INTO {self.tables['summary_statistics']}
        (symbol, timestamp, time_window, price_open, price_close, price_change_pct,
        volume, sentiment_avg, sentiment_std, sentiment_min, sentiment_max,
        news_count, social_count, correlation)
        VALUES
        (%(symbol)s, %(timestamp)s, %(time_window)s, %(price_open)s, %(price_close)s,
        %(price_change_pct)s, %(volume)s, %(sentiment_avg)s, %(sentiment_std)s,
        %(sentiment_min)s, %(sentiment_max)s, %(news_count)s, %(social_count)s,
        %(correlation)s)
        """
        
        try:
            self.execute_query(query, stats_data, fetch=False)
            return True
        except Exception as e:
            logger.error(f"Error inserting summary statistics: {e}")
            return False
    
    def get_summary_statistics(self, symbol=None, time_window=None, start_date=None, end_date=None, limit=100):
        """
        Get summary statistics from the database.
        
        Args:
            symbol (str): Stock ticker symbol
            time_window (str): Time window ('1h', '4h', '1d', '1w')
            start_date (datetime): Start date for filtering
            end_date (datetime): End date for filtering
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of summary statistics records
        """
        query = f"SELECT * FROM {self.tables['summary_statistics']}"
        conditions = []
        params = {}
        
        if symbol:
            conditions.append("symbol = %(symbol)s")
            params['symbol'] = symbol
        
        if time_window:
            conditions.append("time_window = %(time_window)s")
            params['time_window'] = time_window
        
        if start_date:
            conditions.append("timestamp >= %(start_date)s")
            params['start_date'] = start_date
        
        if end_date:
            conditions.append("timestamp <= %(end_date)s")
            params['end_date'] = end_date
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT %(limit)s"
            params['limit'] = limit
        
        try:
            result = self.execute_query(query, params)
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Error getting summary statistics: {e}")
            return []

    def initialize_database(self):
        """
        Initialize the database by executing the create_tables.sql script.
        This method should be called once when the application starts.
        
        Returns:
            bool: Success status
        """
        try:
            # Get the path to the create_tables.sql file
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'create_tables.sql')
            
            # Read the SQL script
            with open(script_path, 'r') as f:
                sql_script = f.read()
            
            # Execute the script
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Execute each statement in the script
            cursor.execute(sql_script)
            conn.commit()
            
            logger.info("Database tables initialized successfully")
            cursor.close()
            self._return_connection(conn)
            return True
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            return False

# Example usage
if __name__ == "__main__":
    db = DatabaseConnector()
    
    # Initialize database tables
    db.initialize_database()
    
    # Test connection
    try:
        # Simple query to test connection
        result = db.execute_query("SELECT 1")
        print("Database connection successful!")
    except Exception as e:
        print(f"Database connection failed: {e}")