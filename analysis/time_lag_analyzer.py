"""
Time lag analysis between sentiment changes and price movements.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import ccf
import logging
import datetime

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_connector import DatabaseConnector
from config.api_config import TARGET_STOCKS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TimeLagAnalyzer:
    """
    Class for analyzing time lags between sentiment and price movements.
    """
    
    def __init__(self):
        """Initialize the time lag analyzer."""
        self.db = DatabaseConnector()
    
    def _prepare_time_series(self, symbol, start_date, end_date, window_size):
        """
        Prepare time series data for lag analysis.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for analysis
            end_date (datetime): End date for analysis
            window_size (int): Window size in minutes
            
        Returns:
            tuple: Tuple containing (price_series, sentiment_series, timestamps)
        """
        # Get stock price data
        price_data = self.db.get_stock_prices(
            symbol=symbol, 
            start_date=start_date, 
            end_date=end_date
        )
        
        # Get sentiment data
        sentiment_data = self.db.get_sentiment_scores(
            symbol=symbol, 
            start_date=start_date, 
            end_date=end_date
        )
        
        if not price_data or not sentiment_data:
            logger.warning(f"Insufficient data for {symbol} in the specified time range")
            return None, None, None
        
        # Convert to DataFrames
        prices_df = pd.DataFrame(price_data)
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Ensure timestamp columns are datetime type
        prices_df['timestamp'] = pd.to_datetime(prices_df['timestamp'])
        sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
        
        # Create time window buckets
        prices_df['window'] = prices_df['timestamp'].dt.floor(f'{window_size}min')
        sentiment_df['window'] = sentiment_df['timestamp'].dt.floor(f'{window_size}min')
        
        # Aggregate data by time windows
        price_windows = prices_df.groupby('window').agg({
            'open': 'first',
            'close': 'last'
        }).reset_index()
        
        sentiment_windows = sentiment_df.groupby('window').agg({
            'compound_score': 'mean'
        }).reset_index()
        
        # Calculate price changes
        if 'open' in price_windows.columns and 'close' in price_windows.columns:
            price_windows['price_change'] = (price_windows['close'] - price_windows['open']) / price_windows['open'] * 100
        else:
            logger.warning(f"Missing required price columns for {symbol}")
            return None, None, None
        
        # Create a full timestamp range to ensure all intervals are included
        full_range = pd.date_range(start=start_date, end=end_date, freq=f'{window_size}min')
        full_df = pd.DataFrame({'window': full_range})
        
        # Merge with full range to ensure all time periods are included
        price_windows = pd.merge(full_df, price_windows, on='window', how='left')
        sentiment_windows = pd.merge(full_df, sentiment_windows, on='window', how='left')
        
        # Convert to numeric before interpolation to avoid object type warning
        price_windows['price_change'] = pd.to_numeric(price_windows['price_change'], errors='coerce')
        sentiment_windows['compound_score'] = pd.to_numeric(sentiment_windows['compound_score'], errors='coerce')
        
        # Interpolate missing values (linear interpolation)
        price_windows['price_change'] = price_windows['price_change'].interpolate(method='linear')
        sentiment_windows['compound_score'] = sentiment_windows['compound_score'].interpolate(method='linear')
        
        # Extract timestamps and series
        timestamps = full_range
        price_series = price_windows['price_change'].values
        sentiment_series = sentiment_windows['compound_score'].values
        
        # Ensure arrays are numeric
        price_series = np.array(price_series, dtype=float)
        sentiment_series = np.array(sentiment_series, dtype=float)
        
        # Handle any remaining NaN values
        price_series = np.nan_to_num(price_series)
        sentiment_series = np.nan_to_num(sentiment_series)
        
        return price_series, sentiment_series, timestamps
    
    def analyze_lag_correlation(self, symbol, start_date, end_date, window_size=60, max_lag=10):
        """
        Analyze time-lagged correlations between sentiment and price movements.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for analysis
            end_date (datetime): End date for analysis
            window_size (int): Window size in minutes
            max_lag (int): Maximum number of lag periods to analyze
            
        Returns:
            dict: Dictionary with lag analysis results
        """
        price_series, sentiment_series, timestamps = self._prepare_time_series(
            symbol, start_date, end_date, window_size
        )
        
        if price_series is None or sentiment_series is None:
            logger.warning(f"Could not prepare time series for {symbol}")
            return {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'window_size': window_size,
                'lag_correlations': [],
                'best_lag': None,
                'best_correlation': None
            }
        
        # Calculate cross-correlation function
        try:
            cross_corr = ccf(sentiment_series, price_series, adjusted=False, method='direct')
        except Exception as e:
            logger.warning(f"Error calculating cross-correlation: {e}")
            cross_corr = []
        
        # Extract correlations for the lags we're interested in
        mid_point = len(cross_corr) // 2 if len(cross_corr) > 0 else 0
        lag_correlations = []
        
        for lag in range(-max_lag, max_lag + 1):
            corr_idx = mid_point + lag
            if 0 <= corr_idx < len(cross_corr):
                lag_correlations.append({
                    'lag': lag,
                    'lag_minutes': lag * window_size,
                    'correlation': float(cross_corr[corr_idx])
                })
        
        # Alternatively, calculate correlations with shifted series
        lagged_correlations = []
        for lag in range(-max_lag, max_lag + 1):
            try:
                if lag < 0:
                    # Sentiment leads price (shift price forward)
                    if abs(lag) >= len(price_series):
                        # Skip if lag is larger than series length
                        continue
                        
                    # Calculate correlation on truncated series of equal length
                    segment_length = min(len(sentiment_series) - abs(lag), len(price_series) - abs(lag))
                    if segment_length < 3:  # Need at least 3 points for correlation
                        continue
                        
                    s_segment = sentiment_series[:segment_length].astype(float)
                    p_segment = price_series[abs(lag):abs(lag)+segment_length].astype(float)
                    
                    if len(s_segment) != len(p_segment):
                        logger.warning(f"Segment lengths don't match for lag {lag}: {len(s_segment)} vs {len(p_segment)}")
                        continue
                        
                    # Check for non-finite values
                    if not np.isfinite(s_segment).all() or not np.isfinite(p_segment).all():
                        logger.warning(f"Non-finite values found in segments for lag {lag}")
                        continue
                        
                    corr, p_value = pearsonr(s_segment, p_segment)
                    
                elif lag > 0:
                    # Price leads sentiment (shift sentiment forward)
                    if lag >= len(sentiment_series):
                        # Skip if lag is larger than series length
                        continue
                        
                    # Calculate correlation on truncated series of equal length
                    segment_length = min(len(sentiment_series) - lag, len(price_series) - lag)
                    if segment_length < 3:  # Need at least 3 points for correlation
                        continue
                        
                    s_segment = sentiment_series[lag:lag+segment_length].astype(float)
                    p_segment = price_series[:segment_length].astype(float)
                    
                    if len(s_segment) != len(p_segment):
                        logger.warning(f"Segment lengths don't match for lag {lag}: {len(s_segment)} vs {len(p_segment)}")
                        continue
                    
                    # Check for non-finite values
                    if not np.isfinite(s_segment).all() or not np.isfinite(p_segment).all():
                        logger.warning(f"Non-finite values found in segments for lag {lag}")
                        continue
                        
                    corr, p_value = pearsonr(s_segment, p_segment)
                    
                else:
                    # No lag
                    segment_length = min(len(sentiment_series), len(price_series))
                    s_segment = sentiment_series[:segment_length].astype(float)
                    p_segment = price_series[:segment_length].astype(float)
                    
                    # Check for non-finite values
                    if not np.isfinite(s_segment).all() or not np.isfinite(p_segment).all():
                        logger.warning(f"Non-finite values found in segments for lag {lag}")
                        continue
                        
                    corr, p_value = pearsonr(s_segment, p_segment)
                
                lagged_correlations.append({
                    'lag': lag,
                    'lag_minutes': lag * window_size,
                    'correlation': float(corr),
                    'p_value': float(p_value)
                })
                
            except Exception as e:
                logger.error(f"Error calculating correlation for lag {lag}: {e}")
                # Continue with next lag despite errors
        
        # Find the best lag
        if not lagged_correlations:
            return {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'window_size': window_size,
                'data_points': len(price_series),
                'lag_correlations': [],
                'best_lag': None,
                'best_lag_minutes': None,
                'best_correlation': None,
                'best_p_value': None
            }
            
        best_lag = max(lagged_correlations, key=lambda x: abs(x['correlation']))
        
        results = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'window_size': window_size,
            'data_points': len(price_series),
            'lag_correlations': lagged_correlations,
            'best_lag': best_lag['lag'],
            'best_lag_minutes': best_lag['lag_minutes'],
            'best_correlation': best_lag['correlation'],
            'best_p_value': best_lag['p_value']
        }
        
        return results
    
    def analyze_all_stocks(self, days=7, window_size=60, max_lag=10):
        """
        Analyze time-lagged correlations for all target stocks.
        
        Args:
            days (int): Number of days to analyze
            window_size (int): Window size in minutes
            max_lag (int): Maximum number of lag periods to analyze
            
        Returns:
            dict: Dictionary with lag analysis results for all stocks
        """
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        all_results = {}
        
        for stock in TARGET_STOCKS:
            symbol = stock['symbol']
            logger.info(f"Analyzing time lag for {symbol}")
            
            try:
                lag_results = self.analyze_lag_correlation(
                    symbol, 
                    start_date, 
                    end_date, 
                    window_size,
                    max_lag
                )
                
                all_results[symbol] = lag_results
            except Exception as e:
                logger.error(f"Error analyzing time lag for {symbol}: {e}")
                all_results[symbol] = {
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date,
                    'window_size': window_size,
                    'error': str(e),
                    'lag_correlations': [],
                    'best_lag': None,
                    'best_lag_minutes': None,
                    'best_correlation': None
                }
        
        return all_results
    
    def plot_lag_analysis(self, symbol, start_date, end_date, window_size=60, max_lag=10):
        """
        Plot lag analysis for a specific stock.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for analysis
            end_date (datetime): End date for analysis
            window_size (int): Window size in minutes
            max_lag (int): Maximum number of lag periods to analyze
            
        Returns:
            matplotlib.figure.Figure: Matplotlib figure with lag analysis plots
        """
        # Get lag analysis results
        results = self.analyze_lag_correlation(
            symbol, start_date, end_date, window_size, max_lag
        )
        
        if not results['lag_correlations']:
            logger.warning(f"No lag correlations available for {symbol}")
            return None
        
        # Extract data for plotting
        lags = [lc['lag'] for lc in results['lag_correlations']]
        correlations = [lc['correlation'] for lc in results['lag_correlations']]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Plot 1: Correlation vs. Lag
        ax1.bar(lags, correlations, alpha=0.7, color='blue')
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.axvline(x=0, color='r', linestyle='--')
        ax1.set_xlabel('Lag (time windows)')
        ax1.set_ylabel('Correlation')
        ax1.set_title(f'Lag Analysis for {symbol}: Best Lag = {results["best_lag"]} windows ({results["best_lag_minutes"]} minutes)')
        ax1.grid(True)
        
        # Highlight best lag
        if results['best_lag'] is not None and results['best_lag'] in lags:
            best_idx = lags.index(results['best_lag'])
            ax1.bar([lags[best_idx]], [correlations[best_idx]], color='green', alpha=0.7)
        
        # Plot 2: Time Series
        price_series, sentiment_series, timestamps = self._prepare_time_series(
            symbol, start_date, end_date, window_size
        )
        
        if timestamps is not None:
            ax2.set_title(f'{symbol} Price Change vs. Sentiment')
            ax2.plot(timestamps, price_series, 'b-', label='Price Change (%)')
            
            # Plot sentiment with a different scale
            ax2_twin = ax2.twinx()
            ax2_twin.plot(timestamps, sentiment_series, 'g-', label='Sentiment')
            
            # Add horizontal lines at zero
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2_twin.axhline(y=0, color='r', linestyle='--')
            
            # Add legends
            ax2.set_ylabel('Price Change (%)', color='b')
            ax2_twin.set_ylabel('Sentiment Score', color='g')
            
            # Create a common legend
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            ax2.grid(True)
        
        # Set common title
        plt.suptitle(f'{symbol} Time Lag Analysis ({window_size}min windows)', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def generate_lag_report(self, days=7, window_size=60, max_lag=10):
        """
        Generate a report on time-lagged correlations.
        
        Args:
            days (int): Number of days to analyze
            window_size (int): Window size in minutes
            max_lag (int): Maximum number of lag periods to analyze
            
        Returns:
            str: Markdown-formatted report
        """
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Get lag analysis for all stocks
        all_results = self.analyze_all_stocks(days, window_size, max_lag)
        
        # Format into a report
        report = [
            f"# Sentiment-Price Time Lag Analysis Report",
            f"**Date Range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            f"**Window Size:** {window_size} minutes",
            f"**Maximum Lag:** {max_lag} windows ({max_lag * window_size} minutes)",
            f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary of Findings",
            "The following table shows the best time lag between sentiment and price changes:",
            "\n| Symbol | Best Lag (windows) | Best Lag (minutes) | Correlation | p-value | Interpretation |",
            "|--------|--------------------|--------------------|-------------|---------|----------------|"
        ]
        
        for symbol, result in all_results.items():
            if result.get('best_lag') is not None:
                interpretation = ""
                if result['best_lag'] < 0:
                    interpretation = f"Sentiment leads price by {abs(result['best_lag_minutes'])} minutes"
                elif result['best_lag'] > 0:
                    interpretation = f"Price leads sentiment by {result['best_lag_minutes']} minutes"
                else:
                    interpretation = "Contemporaneous relationship"
                
                p_value = result.get('best_p_value', "N/A")
                p_value_str = f"{p_value:.4f}" if isinstance(p_value, (int, float)) else p_value
                
                report.append(
                    f"| {symbol} | "
                    f"{result['best_lag']} | "
                    f"{result['best_lag_minutes']} | "
                    f"{result['best_correlation']:.4f} | "
                    f"{p_value_str} | "
                    f"{interpretation} |"
                )
        
        report.extend([
            "\n## Interpretation",
            "- **Negative Lag:** Sentiment changes tend to precede price changes (predictive)",
            "- **Positive Lag:** Price changes tend to precede sentiment changes (reactive)",
            "- **Zero Lag:** Sentiment and price changes occur simultaneously",
            "- **Strong Correlation:** Indicates a reliable relationship at the given lag",
            "- **p-value < 0.05:** Correlation is statistically significant",
            "\n## Detailed Analysis",
        ])
        
        # Sort by absolute correlation
        sorted_results = sorted(
            [r for r in all_results.values() if r.get('best_correlation') is not None],
            key=lambda x: abs(x.get('best_correlation', 0)),
            reverse=True
        )
        
        for result in sorted_results[:5]:  # Top 5 strongest correlations
            if result.get('best_correlation') is not None:
                corr_strength = "strong" if abs(result['best_correlation']) > 0.5 else "moderate" if abs(result['best_correlation']) > 0.3 else "weak"
                corr_direction = "positive" if result['best_correlation'] > 0 else "negative"
                lag_direction = "negative" if result['best_lag'] < 0 else "positive" if result['best_lag'] > 0 else "zero"
                significance = "statistically significant" if result.get('best_p_value', 1) < 0.05 else "not statistically significant"
                
                interpretation = ""
                if result['best_lag'] < 0:
                    interpretation = (
                        f"sentiment changes precede price changes by {abs(result['best_lag_minutes'])} minutes. "
                        f"{'A positive sentiment change tends to be followed by a price increase' if result['best_correlation'] > 0 else 'A positive sentiment change tends to be followed by a price decrease'}"
                    )
                elif result['best_lag'] > 0:
                    interpretation = (
                        f"price changes precede sentiment changes by {result['best_lag_minutes']} minutes. "
                        f"{'A price increase tends to be followed by more positive sentiment' if result['best_correlation'] > 0 else 'A price increase tends to be followed by more negative sentiment'}"
                    )
                else:
                    interpretation = (
                        f"sentiment and price changes occur simultaneously. "
                        f"{'Higher sentiment scores are associated with price increases' if result['best_correlation'] > 0 else 'Higher sentiment scores are associated with price decreases'}"
                    )
                
                report.extend([
                    f"\n### {result['symbol']}",
                    f"- **Best Lag:** {result['best_lag']} windows ({result['best_lag_minutes']} minutes)",
                    f"- **Correlation at Best Lag:** {result['best_correlation']:.4f} ({corr_strength} {corr_direction})",
                    f"- **Statistical Significance:** p-value = {result.get('best_p_value', 'N/A')} ({significance})",
                    f"- **Data Points:** {result['data_points']}",
                    f"\n**Interpretation:** The {lag_direction} lag indicates that {interpretation}. This relationship is {significance}."
                ])
        
        return "\n".join(report)

if __name__ == "__main__":
    analyzer = TimeLagAnalyzer()
    
    # Example usage
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=7)
    
    # Analyze lag for a specific stock
    results = analyzer.analyze_lag_correlation('AAPL', start_date, end_date)
    
    print(f"AAPL Best Lag: {results['best_lag']} windows ({results['best_lag_minutes']} minutes)")
    print(f"Correlation at Best Lag: {results['best_correlation']:.4f} (p-value: {results.get('best_p_value', 'N/A')})")
    
    # Generate and save plot
    fig = analyzer.plot_lag_analysis('AAPL', start_date, end_date)
    if fig:
        fig.savefig('aapl_lag_analysis.png')
        print("Plot saved as aapl_lag_analysis.png")
    
    # Generate report
    report = analyzer.generate_lag_report(days=7)
    with open('lag_analysis_report.md', 'w') as f:
        f.write(report)
    print("Report saved as lag_analysis_report.md")