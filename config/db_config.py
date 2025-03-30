"""
Database configuration settings.
"""

# PostgreSQL connection settings
POSTGRES_CONFIG = {
    "host": "localhost",
    "database": "stock_sentiment_db",
    "user": "postgres",
    "password": "admin",
    "port": 5432
}

# Table names
TABLES = {
    "stock_prices": "stock_prices",
    "news_articles": "news_articles",
    "social_media_posts": "social_media_posts",
    "sentiment_scores": "sentiment_scores",
    "entities": "entities",
    "correlations": "correlations"
}

# Database batch settings
BATCH_SIZE = 100  # Number of records to insert at once
MAX_RETRIES = 3   # Maximum number of connection retry attempts
RETRY_DELAY = 5   # Seconds to wait between retry attempts