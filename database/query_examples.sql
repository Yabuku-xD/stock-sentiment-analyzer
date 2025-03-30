-- Example SQL queries for analyzing stock sentiment data

-- 1. Basic Queries

-- Get the latest stock prices for all symbols
SELECT symbol, 
       timestamp, 
       open, 
       high, 
       low, 
       close, 
       volume
FROM stock_prices
WHERE timestamp > NOW() - INTERVAL '24 hours'
ORDER BY symbol, timestamp DESC;

-- Get the latest sentiment scores by symbol
SELECT 
    symbol, 
    AVG(compound_score) as avg_sentiment,
    COUNT(*) as data_points
FROM sentiment_scores
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY symbol
ORDER BY avg_sentiment DESC;

-- 2. Advanced Queries

-- Calculate hourly average sentiment vs. price changes
WITH hourly_sentiment AS (
    SELECT 
        symbol,
        DATE_TRUNC('hour', timestamp) as hour,
        AVG(compound_score) as avg_sentiment,
        COUNT(*) as sentiment_count
    FROM sentiment_scores
    WHERE timestamp > NOW() - INTERVAL '7 days'
    GROUP BY symbol, DATE_TRUNC('hour', timestamp)
),
hourly_prices AS (
    SELECT 
        symbol,
        DATE_TRUNC('hour', timestamp) as hour,
        FIRST_VALUE(open) OVER (PARTITION BY symbol, DATE_TRUNC('hour', timestamp) ORDER BY timestamp) as hour_open,
        LAST_VALUE(close) OVER (PARTITION BY symbol, DATE_TRUNC('hour', timestamp) ORDER BY timestamp) as hour_close,
        COUNT(*) as price_count
    FROM stock_prices
    WHERE timestamp > NOW() - INTERVAL '7 days'
    GROUP BY symbol, DATE_TRUNC('hour', timestamp), open, close, timestamp
)
SELECT 
    p.symbol,
    p.hour,
    p.hour_open,
    p.hour_close,
    ((p.hour_close - p.hour_open) / p.hour_open * 100) as price_change_pct,
    s.avg_sentiment,
    s.sentiment_count
FROM hourly_prices p
JOIN hourly_sentiment s ON p.symbol = s.symbol AND p.hour = s.hour
ORDER BY p.symbol, p.hour;

-- Find news with the most extreme sentiment for each symbol
WITH ranked_news AS (
    SELECT 
        n.symbol,
        n.headline,
        n.url,
        n.published_at,
        s.compound_score,
        s.sentiment_label,
        ROW_NUMBER() OVER (PARTITION BY n.symbol, s.sentiment_label ORDER BY ABS(s.compound_score) DESC) as rank
    FROM news_articles n
    JOIN sentiment_scores s ON n.news_id = s.reference_id
    WHERE n.published_at > NOW() - INTERVAL '7 days'
)
SELECT 
    symbol,
    headline,
    url,
    published_at,
    compound_score,
    sentiment_label
FROM ranked_news
WHERE rank = 1 AND sentiment_label IN ('positive', 'negative')
ORDER BY symbol, sentiment_label;

-- 3. Time-based Analysis

-- Calculate sentiment momentum (rate of change over time)
WITH daily_sentiment AS (
    SELECT 
        symbol,
        DATE_TRUNC('day', timestamp) as day,
        AVG(compound_score) as avg_sentiment
    FROM sentiment_scores
    WHERE timestamp > NOW() - INTERVAL '30 days'
    GROUP BY symbol, DATE_TRUNC('day', timestamp)
),
sentiment_change AS (
    SELECT 
        symbol,
        day,
        avg_sentiment,
        LAG(avg_sentiment, 1) OVER (PARTITION BY symbol ORDER BY day) as prev_day_sentiment
    FROM daily_sentiment
)
SELECT 
    symbol,
    day,
    avg_sentiment,
    prev_day_sentiment,
    (avg_sentiment - prev_day_sentiment) as sentiment_change,
    CASE 
        WHEN (avg_sentiment - prev_day_sentiment) > 0 THEN 'improving'
        WHEN (avg_sentiment - prev_day_sentiment) < 0 THEN 'declining'
        ELSE 'stable'
    END as sentiment_momentum
FROM sentiment_change
WHERE prev_day_sentiment IS NOT NULL
ORDER BY symbol, day;

-- 4. Correlation Analysis

-- Calculate correlation between sentiment and price changes
WITH daily_sentiment AS (
    SELECT 
        symbol,
        DATE_TRUNC('day', timestamp) as day,
        AVG(compound_score) as avg_sentiment
    FROM sentiment_scores
    WHERE timestamp > NOW() - INTERVAL '30 days'
    GROUP BY symbol, DATE_TRUNC('day', timestamp)
),
daily_prices AS (
    SELECT 
        symbol,
        DATE_TRUNC('day', timestamp) as day,
        FIRST_VALUE(open) OVER (PARTITION BY symbol, DATE_TRUNC('day', timestamp) ORDER BY timestamp) as day_open,
        LAST_VALUE(close) OVER (PARTITION BY symbol, DATE_TRUNC('day', timestamp) ORDER BY timestamp) as day_close
    FROM stock_prices
    WHERE timestamp > NOW() - INTERVAL '30 days'
    GROUP BY symbol, DATE_TRUNC('day', timestamp), open, close, timestamp
),
daily_changes AS (
    SELECT 
        p.symbol,
        p.day,
        ((p.day_close - p.day_open) / p.day_open * 100) as price_change_pct,
        s.avg_sentiment
    FROM daily_prices p
    JOIN daily_sentiment s ON p.symbol = s.symbol AND p.day = s.day
)
SELECT 
    symbol,
    CORR(price_change_pct, avg_sentiment) as correlation,
    COUNT(*) as data_points
FROM daily_changes
GROUP BY symbol
ORDER BY ABS(correlation) DESC;

-- 5. Entity Analysis

-- Find most frequently mentioned entities in positive news
SELECT 
    e.entity_value,
    e.entity_type,
    COUNT(*) as mention_count,
    AVG(s.compound_score) as avg_sentiment
FROM entities e
JOIN sentiment_scores s ON e.reference_id = s.reference_id
WHERE 
    e.entity_type IN ('ticker', 'company') AND
    s.sentiment_label = 'positive' AND
    e.timestamp > NOW() - INTERVAL '7 days'
GROUP BY e.entity_value, e.entity_type
HAVING COUNT(*) > 5
ORDER BY avg_sentiment DESC, mention_count DESC;

-- 6. Source Analysis

-- Compare sentiment across different news sources
SELECT 
    s.source,
    s.symbol,
    AVG(s.compound_score) as avg_sentiment,
    COUNT(*) as article_count
FROM sentiment_scores s
WHERE 
    s.content_type = 'news' AND
    s.timestamp > NOW() - INTERVAL '30 days'
GROUP BY s.source, s.symbol
HAVING COUNT(*) > 10
ORDER BY s.symbol, avg_sentiment DESC;

-- 7. Time Lag Analysis

-- Analyze if sentiment changes precede price changes (1-day lag)
WITH daily_sentiment AS (
    SELECT 
        symbol,
        DATE_TRUNC('day', timestamp) as day,
        AVG(compound_score) as avg_sentiment
    FROM sentiment_scores
    WHERE timestamp > NOW() - INTERVAL '90 days'
    GROUP BY symbol, DATE_TRUNC('day', timestamp)
),
daily_prices AS (
    SELECT 
        symbol,
        DATE_TRUNC('day', timestamp) as day,
        AVG(((close - open) / open) * 100) as avg_price_change_pct
    FROM stock_prices
    WHERE timestamp > NOW() - INTERVAL '90 days'
    GROUP BY symbol, DATE_TRUNC('day', timestamp)
),
lagged_data AS (
    SELECT 
        s.symbol,
        s.day,
        s.avg_sentiment,
        p.avg_price_change_pct as same_day_price_change,
        LEAD(p.avg_price_change_pct, 1) OVER (PARTITION BY s.symbol ORDER BY s.day) as next_day_price_change
    FROM daily_sentiment s
    LEFT JOIN daily_prices p ON s.symbol = p.symbol AND s.day = p.day
)
SELECT 
    symbol,
    CORR(avg_sentiment, same_day_price_change) as same_day_correlation,
    CORR(avg_sentiment, next_day_price_change) as next_day_correlation,
    COUNT(*) as data_points
FROM lagged_data
WHERE next_day_price_change IS NOT NULL
GROUP BY symbol
HAVING COUNT(*) > 20
ORDER BY ABS(next_day_correlation) DESC;

-- 8. Summary Statistics for Dashboard

-- Create summary statistics for different time windows
INSERT INTO summary_statistics 
(symbol, timestamp, time_window, price_open, price_close, price_change_pct, 
 volume, sentiment_avg, sentiment_std, sentiment_min, sentiment_max, 
 news_count, social_count, correlation)
WITH price_stats AS (
    SELECT 
        symbol,
        FIRST_VALUE(open) OVER (PARTITION BY symbol ORDER BY timestamp) as first_open,
        LAST_VALUE(close) OVER (PARTITION BY symbol ORDER BY timestamp) as last_close,
        SUM(volume) as total_volume
    FROM stock_prices
    WHERE timestamp > NOW() - INTERVAL '24 hours'
    GROUP BY symbol, timestamp, open, close, volume
),
sentiment_stats AS (
    SELECT 
        symbol,
        AVG(compound_score) as avg_sentiment,
        STDDEV(compound_score) as std_sentiment,
        MIN(compound_score) as min_sentiment,
        MAX(compound_score) as max_sentiment,
        COUNT(CASE WHEN content_type = 'news' THEN 1 END) as news_count,
        COUNT(CASE WHEN content_type = 'social' THEN 1 END) as social_count
    FROM sentiment_scores
    WHERE timestamp > NOW() - INTERVAL '24 hours'
    GROUP BY symbol
),
correlation_calc AS (
    SELECT 
        p.symbol,
        DATE_TRUNC('hour', p.timestamp) as hour,
        AVG(((p.close - p.open) / p.open) * 100) as price_change,
        AVG(s.compound_score) as sentiment
    FROM stock_prices p
    JOIN sentiment_scores s ON p.symbol = s.symbol 
        AND DATE_TRUNC('hour', p.timestamp) = DATE_TRUNC('hour', s.timestamp)
    WHERE p.timestamp > NOW() - INTERVAL '24 hours'
    GROUP BY p.symbol, DATE_TRUNC('hour', p.timestamp)
),
corr_by_symbol AS (
    SELECT 
        symbol,
        CORR(price_change, sentiment) as correlation
    FROM correlation_calc
    GROUP BY symbol
)
SELECT 
    p.symbol,
    NOW() as timestamp,
    '1d' as time_window,
    p.first_open,
    p.last_close,
    ((p.last_close - p.first_open) / p.first_open * 100) as price_change_pct,
    p.total_volume,
    s.avg_sentiment,
    s.std_sentiment,
    s.min_sentiment,
    s.max_sentiment,
    s.news_count,
    s.social_count,
    c.correlation
FROM price_stats p
JOIN sentiment_stats s ON p.symbol = s.symbol
LEFT JOIN corr_by_symbol c ON p.symbol = c.symbol;