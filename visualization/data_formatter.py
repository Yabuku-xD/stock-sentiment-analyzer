"""
Data formatter module for preparing data for visualization.
"""

import pandas as pd
import numpy as np
import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataFormatter:
    """
    Class for formatting and preparing data for visualization.
    """
    
    @staticmethod
    def format_time_series(df, timestamp_col='timestamp', resample_interval='1min', fill_method='ffill'):
        """
        Format a time series dataframe to have regular intervals.
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            timestamp_col (str): Column name for timestamps
            resample_interval (str): Interval for resampling
            fill_method (str): Method for filling missing values
            
        Returns:
            pandas.DataFrame: Formatted DataFrame
        """
        if df.empty:
            return df
        
        # Ensure timestamp column is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by timestamp
        df = df.sort_values(timestamp_col)
        
        # Set timestamp as index
        df = df.set_index(timestamp_col)
        
        # Resample to regular intervals
        # This will create a regular time series with the specified interval
        df_resampled = df.resample(resample_interval)
        
        # Fill missing values based on method
        if fill_method == 'ffill':
            df_resampled = df_resampled.ffill()
        elif fill_method == 'bfill':
            df_resampled = df_resampled.bfill()
        elif fill_method == 'interpolate':
            df_resampled = df_resampled.interpolate(method='linear')
        else:
            df_resampled = df_resampled.ffill()  # Default to forward fill
        
        # Reset index to move timestamp back to a column
        df_resampled = df_resampled.reset_index()
        
        return df_resampled
    
    @staticmethod
    def align_time_series(df1, df2, timestamp_col='timestamp', resample_interval='1min'):
        """
        Align two time series dataframes to have the same timestamps.
        
        Args:
            df1 (pandas.DataFrame): First DataFrame
            df2 (pandas.DataFrame): Second DataFrame
            timestamp_col (str): Column name for timestamps
            resample_interval (str): Interval for resampling
            
        Returns:
            tuple: Tuple of aligned DataFrames
        """
        if df1.empty or df2.empty:
            return df1, df2
        
        # Ensure timestamp columns are datetime
        df1[timestamp_col] = pd.to_datetime(df1[timestamp_col])
        df2[timestamp_col] = pd.to_datetime(df2[timestamp_col])
        
        # Find common time range
        start_time = max(df1[timestamp_col].min(), df2[timestamp_col].min())
        end_time = min(df1[timestamp_col].max(), df2[timestamp_col].max())
        
        # Filter to common time range
        df1_filtered = df1[(df1[timestamp_col] >= start_time) & (df1[timestamp_col] <= end_time)]
        df2_filtered = df2[(df2[timestamp_col] >= start_time) & (df2[timestamp_col] <= end_time)]
        
        # Format each time series
        df1_formatted = DataFormatter.format_time_series(df1_filtered, timestamp_col, resample_interval)
        df2_formatted = DataFormatter.format_time_series(df2_filtered, timestamp_col, resample_interval)
        
        # Create a common index of timestamps
        all_timestamps = sorted(set(df1_formatted[timestamp_col]).union(set(df2_formatted[timestamp_col])))
        
        # Reindex both dataframes to the common timestamps
        df1_reindexed = df1_formatted.set_index(timestamp_col).reindex(all_timestamps).reset_index()
        df2_reindexed = df2_formatted.set_index(timestamp_col).reindex(all_timestamps).reset_index()
        
        # Rename timestamp column back
        df1_reindexed = df1_reindexed.rename(columns={'index': timestamp_col})
        df2_reindexed = df2_reindexed.rename(columns={'index': timestamp_col})
        
        return df1_reindexed, df2_reindexed
    
    @staticmethod
    def prepare_candlestick_data(price_df, interval='5min'):
        """
        Prepare stock price data for candlestick chart.
        
        Args:
            price_df (pandas.DataFrame): Price DataFrame
            interval (str): Interval for resampling
            
        Returns:
            pandas.DataFrame: Prepared DataFrame for candlestick chart
        """
        if price_df.empty:
            return price_df
        
        # Ensure timestamp column is datetime
        price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
        
        # Sort by timestamp
        price_df = price_df.sort_values('timestamp')
        
        # Set timestamp as index
        price_df = price_df.set_index('timestamp')
        
        # Resample to specified interval
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Only include columns that exist in the dataframe
        valid_cols = [col for col in ohlc_dict.keys() if col in price_df.columns]
        valid_dict = {col: ohlc_dict[col] for col in valid_cols}
        
        # Resample
        price_resampled = price_df[valid_cols].resample(interval).agg(valid_dict)
        
        # Reset index
        price_resampled = price_resampled.reset_index()
        
        return price_resampled
    
    @staticmethod
    def calculate_moving_averages(df, value_col, windows=[5, 10, 20]):
        """
        Calculate moving averages for a time series.
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            value_col (str): Column to calculate moving averages for
            windows (list): List of window sizes
            
        Returns:
            pandas.DataFrame: DataFrame with moving averages
        """
        if df.empty or value_col not in df.columns:
            return df
        
        result_df = df.copy()
        
        for window in windows:
            result_df[f'{value_col}_ma{window}'] = result_df[value_col].rolling(window=window).mean()
        
        return result_df
    
    @staticmethod
    def calculate_correlations(price_df, sentiment_df, price_col='price_change', sentiment_col='compound_score'):
        """
        Calculate correlations between price and sentiment at different time lags.
        
        Args:
            price_df (pandas.DataFrame): Price DataFrame with 'timestamp' column
            sentiment_df (pandas.DataFrame): Sentiment DataFrame with 'timestamp' column
            price_col (str): Column in price_df to use for correlation
            sentiment_col (str): Column in sentiment_df to use for correlation
            
        Returns:
            pandas.DataFrame: DataFrame with correlations at different lags
        """
        # First ensure we have the required columns
        if (price_df.empty or sentiment_df.empty or 
            'timestamp' not in price_df.columns or 'timestamp' not in sentiment_df.columns or
            price_col not in price_df.columns or sentiment_col not in sentiment_df.columns):
            return pd.DataFrame(columns=['lag', 'correlation'])
        
        # Align time series to hourly intervals
        price_hourly = price_df.copy()
        sentiment_hourly = sentiment_df.copy()
        
        price_hourly['hour'] = price_hourly['timestamp'].dt.floor('H')
        sentiment_hourly['hour'] = sentiment_hourly['timestamp'].dt.floor('H')
        
        price_agg = price_hourly.groupby('hour')[price_col].mean().reset_index()
        sentiment_agg = sentiment_hourly.groupby('hour')[sentiment_col].mean().reset_index()
        
        # Merge on hour
        merged = pd.merge(price_agg, sentiment_agg, on='hour', how='inner')
        
        if len(merged) < 3:
            return pd.DataFrame(columns=['lag', 'correlation'])
        
        # Calculate correlations for different lags
        max_lag = min(10, len(merged) // 3)  # Set reasonable max lag
        correlations = []
        
        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                # Sentiment leads price (shift price forward)
                merged[f'price_shifted'] = merged[price_col].shift(lag)
                corr = merged[[sentiment_col, 'price_shifted']].corr().iloc[0, 1]
            elif lag > 0:
                # Price leads sentiment (shift sentiment forward)
                merged[f'sentiment_shifted'] = merged[sentiment_col].shift(lag)
                corr = merged[[price_col, 'sentiment_shifted']].corr().iloc[0, 1]
            else:
                # No lag
                corr = merged[[price_col, sentiment_col]].corr().iloc[0, 1]
            
            correlations.append({
                'lag': lag,
                'correlation': corr if not np.isnan(corr) else 0
            })
        
        return pd.DataFrame(correlations)
    
    @staticmethod
    def aggregate_data_by_time(df, timestamp_col='timestamp', agg_interval='1h', numeric_agg='mean', text_agg='first'):
        """
        Aggregate data by time intervals.
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            timestamp_col (str): Column name for timestamps
            agg_interval (str): Interval for aggregation
            numeric_agg (str): Aggregation method for numeric columns
            text_agg (str): Aggregation method for text columns
            
        Returns:
            pandas.DataFrame: Aggregated DataFrame
        """
        if df.empty or timestamp_col not in df.columns:
            return df
        
        # Ensure timestamp column is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Create time bucket column
        if agg_interval.endswith('h'):
            hours = int(agg_interval[:-1])
            df['time_bucket'] = df[timestamp_col].dt.floor(f'{hours}H')
        elif agg_interval.endswith('min'):
            minutes = int(agg_interval[:-3])
            df['time_bucket'] = df[timestamp_col].dt.floor(f'{minutes}min')
        elif agg_interval.endswith('d'):
            days = int(agg_interval[:-1])
            df['time_bucket'] = df[timestamp_col].dt.floor(f'{days}D')
        else:
            df['time_bucket'] = df[timestamp_col].dt.floor('1H')  # Default to hourly
        
        # Determine aggregation method for each column
        agg_dict = {}
        for col in df.columns:
            if col in [timestamp_col, 'time_bucket']:
                continue
                
            if pd.api.types.is_numeric_dtype(df[col]):
                agg_dict[col] = numeric_agg
            else:
                agg_dict[col] = text_agg
        
        # Aggregate by time bucket
        agg_df = df.groupby('time_bucket').agg(agg_dict).reset_index()
        
        # Rename time_bucket back to timestamp_col
        agg_df = agg_df.rename(columns={'time_bucket': timestamp_col})
        
        return agg_df
    
    @staticmethod
    def format_for_dashboard(price_df, sentiment_df, news_df=None, social_df=None, interval='1h'):
        """
        Format all data for dashboard display.
        
        Args:
            price_df (pandas.DataFrame): Price DataFrame
            sentiment_df (pandas.DataFrame): Sentiment DataFrame
            news_df (pandas.DataFrame): News DataFrame
            social_df (pandas.DataFrame): Social media DataFrame
            interval (str): Interval for aggregation
            
        Returns:
            dict: Dictionary with formatted data
        """
        result = {}
        
        # Format price data for candlestick chart
        if price_df is not None and not price_df.empty:
            result['price_data'] = DataFormatter.prepare_candlestick_data(price_df, interval)
            
            # Calculate price changes
            if 'close' in result['price_data'].columns and 'open' in result['price_data'].columns:
                result['price_data']['price_change'] = (result['price_data']['close'] - result['price_data']['open']) / result['price_data']['open'] * 100
        
        # Format sentiment data
        if sentiment_df is not None and not sentiment_df.empty:
            result['sentiment_data'] = DataFormatter.aggregate_data_by_time(sentiment_df, 'timestamp', interval)
            
            # Calculate moving averages
            if 'compound_score' in result['sentiment_data'].columns:
                result['sentiment_data'] = DataFormatter.calculate_moving_averages(
                    result['sentiment_data'], 'compound_score', [5, 10]
                )
        
        # Format news data
        if news_df is not None and not news_df.empty:
            result['news_data'] = news_df.sort_values('published_at', ascending=False) if 'published_at' in news_df.columns else news_df
        
        # Format social media data
        if social_df is not None and not social_df.empty:
            result['social_data'] = social_df.sort_values('created_at', ascending=False) if 'created_at' in social_df.columns else social_df
        
        # Calculate correlations
        if 'price_data' in result and 'sentiment_data' in result:
            price_df_for_corr = result['price_data'].copy()
            sentiment_df_for_corr = result['sentiment_data'].copy()
            
            if 'price_change' in price_df_for_corr.columns and 'compound_score' in sentiment_df_for_corr.columns:
                result['correlation_data'] = DataFormatter.calculate_correlations(
                    price_df_for_corr, sentiment_df_for_corr
                )
        
        return result

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
    
    price_data = {
        'timestamp': dates,
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(105, 5, 100),
        'low': np.random.normal(95, 5, 100),
        'close': np.random.normal(102, 5, 100),
        'volume': np.random.randint(1000, 10000, 100)
    }
    
    sentiment_data = {
        'timestamp': dates,
        'compound_score': np.random.normal(0.2, 0.5, 100),
        'positive_score': np.random.normal(0.5, 0.2, 100),
        'negative_score': np.random.normal(0.3, 0.2, 100),
        'neutral_score': np.random.normal(0.2, 0.1, 100)
    }
    
    price_df = pd.DataFrame(price_data)
    sentiment_df = pd.DataFrame(sentiment_data)
    
    # Format time series
    formatted_price = DataFormatter.format_time_series(price_df)
    print(f"Formatted price data shape: {formatted_price.shape}")
    
    # Align time series
    aligned_price, aligned_sentiment = DataFormatter.align_time_series(price_df, sentiment_df)
    print(f"Aligned price data shape: {aligned_price.shape}")
    print(f"Aligned sentiment data shape: {aligned_sentiment.shape}")
    
    # Prepare candlestick data
    candlestick_data = DataFormatter.prepare_candlestick_data(price_df, interval='1h')
    print(f"Candlestick data shape: {candlestick_data.shape}")
    
    # Calculate moving averages
    ma_data = DataFormatter.calculate_moving_averages(sentiment_df, 'compound_score')
    print(f"Moving average columns: {[col for col in ma_data.columns if col.startswith('compound_score_ma')]}")
    
    # Calculate correlations
    price_df['price_change'] = price_df['close'] - price_df['open']
    correlations = DataFormatter.calculate_correlations(price_df, sentiment_df)
    print(f"Correlation data shape: {correlations.shape}")
    print(f"Best lag: {correlations.loc[correlations['correlation'].abs().idxmax()]}")
    
    # Format all data for dashboard
    dashboard_data = DataFormatter.format_for_dashboard(price_df, sentiment_df)
    print(f"Dashboard data keys: {dashboard_data.keys()}")