import logging
from database.db_connector import DatabaseConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_database_initialization():
    """Test database initialization and verify tables exist."""
    logger.info("Creating database connector...")
    db = DatabaseConnector()
    
    logger.info("Initializing database tables...")
    success = db.initialize_database()
    logger.info(f"Database initialization success: {success}")
    
    # Test if tables exist by running a simple query
    try:
        # Test news_articles table
        result = db.execute_query("SELECT COUNT(*) FROM news_articles")
        logger.info(f"news_articles table exists, count: {result[0][0]}")
        
        # Test other tables
        tables = ['stock_prices', 'social_media_posts', 'sentiment_scores', 'entities', 'correlations', 'summary_statistics']
        for table in tables:
            result = db.execute_query(f"SELECT COUNT(*) FROM {table}")
            logger.info(f"{table} table exists, count: {result[0][0]}")
            
        logger.info("All database tables verified successfully!")
        return True
    except Exception as e:
        logger.error(f"Error verifying tables: {e}")
        return False

if __name__ == "__main__":
    test_database_initialization()