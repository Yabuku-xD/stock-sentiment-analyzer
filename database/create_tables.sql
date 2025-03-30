-- Create database (run this separately if needed)
-- CREATE DATABASE stock_sentiment_db;

-- Stock Prices Table
CREATE TABLE IF NOT EXISTS stock_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open NUMERIC(10,2) NOT NULL,
    high NUMERIC(10,2) NOT NULL,
    low NUMERIC(10,2) NOT NULL,
    close NUMERIC(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    source VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for stock_prices
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices(symbol);
CREATE INDEX IF NOT EXISTS idx_stock_prices_timestamp ON stock_prices(timestamp);
CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol_timestamp ON stock_prices(symbol, timestamp);

-- News Articles Table
CREATE TABLE IF NOT EXISTS news_articles (
    id SERIAL PRIMARY KEY,
    news_id VARCHAR(100) NOT NULL UNIQUE,
    source VARCHAR(100) NOT NULL,
    symbol VARCHAR(10),
    headline TEXT NOT NULL,
    summary TEXT,
    url TEXT,
    published_at TIMESTAMP,
    collected_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for news_articles
CREATE INDEX IF NOT EXISTS idx_news_articles_symbol ON news_articles(symbol);
CREATE INDEX IF NOT EXISTS idx_news_articles_published_at ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_news_articles_news_id ON news_articles(news_id);

-- Social Media Posts Table
CREATE TABLE IF NOT EXISTS social_media_posts (
    id SERIAL PRIMARY KEY,
    post_id VARCHAR(100) NOT NULL UNIQUE,
    platform VARCHAR(50) NOT NULL,
    subreddit VARCHAR(50),
    symbol VARCHAR(10),
    title TEXT,
    content TEXT,
    url TEXT,
    user_name VARCHAR(100),
    upvotes INT,
    comments INT,
    created_at TIMESTAMP,
    collected_at TIMESTAMP NOT NULL,
    created_at_db TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for social_media_posts
CREATE INDEX IF NOT EXISTS idx_social_media_posts_symbol ON social_media_posts(symbol);
CREATE INDEX IF NOT EXISTS idx_social_media_posts_created_at ON social_media_posts(created_at);
CREATE INDEX IF NOT EXISTS idx_social_media_posts_platform ON social_media_posts(platform);
CREATE INDEX IF NOT EXISTS idx_social_media_posts_post_id ON social_media_posts(post_id);

-- Sentiment Scores Table
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id SERIAL PRIMARY KEY,
    reference_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    source VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    content_type VARCHAR(20) NOT NULL, -- 'news' or 'social'
    compound_score NUMERIC(5,4) NOT NULL,
    positive_score NUMERIC(5,4) NOT NULL,
    negative_score NUMERIC(5,4) NOT NULL,
    neutral_score NUMERIC(5,4) NOT NULL,
    sentiment_label VARCHAR(10) NOT NULL, -- 'positive', 'negative', or 'neutral'
    analyzed_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for sentiment_scores
CREATE INDEX IF NOT EXISTS idx_sentiment_scores_symbol ON sentiment_scores(symbol);
CREATE INDEX IF NOT EXISTS idx_sentiment_scores_timestamp ON sentiment_scores(timestamp);
CREATE INDEX IF NOT EXISTS idx_sentiment_scores_reference_id ON sentiment_scores(reference_id);
CREATE INDEX IF NOT EXISTS idx_sentiment_scores_content_type ON sentiment_scores(content_type);
CREATE INDEX IF NOT EXISTS idx_sentiment_scores_sentiment_label ON sentiment_scores(sentiment_label);

-- Entities Table
CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    reference_id VARCHAR(100) NOT NULL,
    source VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL, -- 'ticker', 'company', 'metric_currency', etc.
    entity_value VARCHAR(100) NOT NULL,
    entity_text TEXT,
    timestamp TIMESTAMP NOT NULL,
    extracted_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for entities
CREATE INDEX IF NOT EXISTS idx_entities_reference_id ON entities(reference_id);
CREATE INDEX IF NOT EXISTS idx_entities_entity_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_entity_value ON entities(entity_value);
CREATE INDEX IF NOT EXISTS idx_entities_timestamp ON entities(timestamp);

-- Correlations Table
CREATE TABLE IF NOT EXISTS correlations (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    window_size INT NOT NULL, -- in minutes
    price_change NUMERIC(7,4) NOT NULL,
    sentiment_change NUMERIC(7,4) NOT NULL,
    correlation_value NUMERIC(5,4) NOT NULL,
    data_points INT NOT NULL,
    calculated_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for correlations
CREATE INDEX IF NOT EXISTS idx_correlations_symbol ON correlations(symbol);
CREATE INDEX IF NOT EXISTS idx_correlations_date_range ON correlations(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_correlations_window_size ON correlations(window_size);

-- Summary Statistics Table (for dashboard)
CREATE TABLE IF NOT EXISTS summary_statistics (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    time_window VARCHAR(20) NOT NULL, -- '1h', '4h', '1d', '1w'
    price_open NUMERIC(10,2),
    price_close NUMERIC(10,2),
    price_change_pct NUMERIC(7,4),
    volume BIGINT,
    sentiment_avg NUMERIC(5,4),
    sentiment_std NUMERIC(5,4),
    sentiment_min NUMERIC(5,4),
    sentiment_max NUMERIC(5,4),
    news_count INT,
    social_count INT,
    correlation NUMERIC(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for summary_statistics
CREATE INDEX IF NOT EXISTS idx_summary_statistics_symbol ON summary_statistics(symbol);
CREATE INDEX IF NOT EXISTS idx_summary_statistics_timestamp ON summary_statistics(timestamp);
CREATE INDEX IF NOT EXISTS idx_summary_statistics_time_window ON summary_statistics(time_window);