"""
Test script for initializing and verifying the database.
"""

import os
import sys
import logging
from pathlib import Path

# Make sure log directory exists
Path("logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/db_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import database connector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from database.db_connector import DatabaseConnector
from config.db_config import TABLES

def test_database_initialization():
    """Test database initialization and verify tables exist."""
    logger.info("Creating database connector...")
    db = DatabaseConnector()
    
    logger.info("Initializing database tables...")
    success = db.initialize_database()
    logger.info(f"Database initialization success: {success}")
    
    # Test if tables exist by checking each one
    tables_verified = 0
    tables_failed = 0
    
    for table_name in TABLES.values():
        try:
            # Check if table exists
            exists = db.check_table_exists(table_name)
            if exists:
                # Try counting records
                result = db.execute_query(f"SELECT COUNT(*) FROM {table_name}")
                count = result[0][0] if result else 0
                logger.info(f"✓ Table '{table_name}' exists, contains {count} records")
                tables_verified += 1
            else:
                logger.error(f"✗ Table '{table_name}' does not exist")
                tables_failed += 1
        except Exception as e:
            logger.error(f"✗ Error verifying table '{table_name}': {e}")
            tables_failed += 1
    
    if tables_failed == 0:
        logger.info(f"✓ All database tables ({tables_verified}) verified successfully!")
        return True
    else:
        logger.error(f"✗ {tables_failed} tables failed verification, {tables_verified} succeeded")
        return False

def test_basic_operations():
    """Test basic database operations."""
    logger.info("Testing basic database operations...")
    db = DatabaseConnector()
    
    # Test 1: Insert stock price
    logger.info("Test 1: Insert a stock price record")
    import datetime
    
    test_price = {
        'symbol': 'TEST',
        'timestamp': datetime.datetime.now(),
        'open': 100.0,
        'high': 105.0,
        'low': 95.0,
        'close': 102.0,
        'volume': 1000,
        'source': 'Test'
    }
    
    try:
        rows = db.insert_stock_prices([test_price])
        logger.info(f"✓ Inserted {rows} stock price record(s)")
    except Exception as e:
        logger.error(f"✗ Failed to insert stock price: {e}")
    
    # Test 2: Query stock price
    logger.info("Test 2: Query stock price record")
    try:
        prices = db.get_stock_prices(symbol='TEST', limit=1)
        if prices:
            logger.info(f"✓ Successfully retrieved {len(prices)} stock price record(s)")
        else:
            logger.warning("⚠ No stock price records found")
    except Exception as e:
        logger.error(f"✗ Failed to query stock prices: {e}")
    
    # Test 3: Insert news article
    logger.info("Test 3: Insert a news article")
    test_news = {
        'news_id': f'test-{datetime.datetime.now().timestamp()}',
        'source': 'Test Source',
        'symbol': 'TEST',
        'headline': 'Test Headline',
        'summary': 'This is a test summary.',
        'url': 'https://example.com',
        'published_at': datetime.datetime.now(),
        'collected_at': datetime.datetime.now()
    }
    
    try:
        rows = db.insert_news_articles([test_news])
        logger.info(f"✓ Inserted {rows} news article(s)")
    except Exception as e:
        logger.error(f"✗ Failed to insert news article: {e}")
    
    logger.info("Basic operations test completed")

if __name__ == "__main__":
    print("\n=== TESTING DATABASE INITIALIZATION ===\n")
    if test_database_initialization():
        print("\n✓ Database initialization test passed!")
        
        print("\n=== TESTING BASIC OPERATIONS ===\n")
        test_basic_operations()
        print("\n✓ Basic operations test completed")
    else:
        print("\n✗ Database initialization test failed!")