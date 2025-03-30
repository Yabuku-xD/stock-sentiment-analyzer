"""
Correlation analysis between sentiment and stock price movements.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
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

class CorrelationAnalyzer:
    """
    Class for analyzing correlations between sentiment and price movements.
    """
    
    def __init__(self):
        """Initialize the correlation analyzer."""
        self.db = DatabaseConnector()
    
    def _get_aligned_data(self, symbol, start_date, end_date, window_size):
        """
        Get sentiment and price data aligned by time windows.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for analysis
            end_date (datetime): End date for analysis
            window_size (int): Window size in minutes
            
        Returns:
            pandas.DataFrame: DataFrame with aligned sentiment and price data
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
            return None
        
        # Convert to DataFrames
        prices_df = pd.DataFrame(price_data)
        sentiment_df = pd.DataFrame(sentiment_data)
        
        # Create time window buckets
        prices_df['window'] = prices_df['timestamp'].dt.floor(f'{window_size}min')
        sentiment_df['window'] = sentiment_df['timestamp'].dt.floor(f'{window_size}min')
        
        # Aggregate data by time windows
        price_windows = prices_df.groupby('window').agg({
            'open': 'first',
            'close': 'last',
            'high': 'max',
            'low': 'min',
            'volume': 'sum'
        }).reset_index()
        
        sentiment_windows = sentiment_df.groupby('window').agg({
            'compound_score': 'mean',
            'positive_score': 'mean',
            'negative_score': 'mean',
            'neutral_score': 'mean',
            'reference_id': 'count'  # Count of sentiment data points
        }).reset_index()
        sentiment_windows.rename(columns={'reference_id': 'data_points'}, inplace=True)
        
        # Calculate price changes
        price_windows['price_change'] = (price_windows['close'] - price_windows['open']) / price_windows['open'] * 100
        
        # Merge data on time windows
        merged_data = pd.merge(price_windows, sentiment_windows, on='window', how='inner')
        
        return merged_data
    
    def calculate_correlation(self, symbol, start_date, end_date, window_size=60):
        """
        Calculate correlation between sentiment and price changes.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for analysis
            end_date (datetime): End date for analysis
            window_size (int): Window size in minutes
            
        Returns:
            dict: Dictionary with correlation metrics
        """
        aligned_data = self._get_aligned_data(symbol, start_date, end_date, window_size)
        
        if aligned_data is None or aligned_data.empty or len(aligned_data) < 3:
            logger.warning(f"Insufficient aligned data points for {symbol}")
            return {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'window_size': window_size,
                'data_points': 0,
                'pearson_correlation': None,
                'spearman_correlation': None,
                'price_change_avg': None,
                'sentiment_avg': None
            }
        
        # Calculate Pearson correlation (linear)
        pearson_corr, pearson_p = pearsonr(
            aligned_data['compound_score'], 
            aligned_data['price_change']
        )
        
        # Calculate Spearman correlation (rank-based, detects monotonic relationships)
        spearman_corr, spearman_p = spearmanr(
            aligned_data['compound_score'], 
            aligned_data['price_change']
        )
        
        results = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'window_size': window_size,
            'data_points': len(aligned_data),
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
            'price_change_avg': aligned_data['price_change'].mean(),
            'price_change_std': aligned_data['price_change'].std(),
            'sentiment_avg': aligned_data['compound_score'].mean(),
            'sentiment_std': aligned_data['compound_score'].std(),
            'aligned_data': aligned_data
        }
        
        # Store correlation results in database
        self._store_correlation(results)
        
        return results
    
    def _store_correlation(self, results):
        """
        Store correlation results in the database.
        
        Args:
            results (dict): Correlation results
        """
        # Prepare data for database
        correlation_data = {
            'symbol': results['symbol'],
            'start_date': results['start_date'],
            'end_date': results['end_date'],
            'window_size': results['window_size'],
            'price_change': results['price_change_avg'],
            'sentiment_change': results['sentiment_avg'],
            'correlation_value': results['pearson_correlation'],
            'data_points': results['data_points'],
            'calculated_at': datetime.datetime.now()
        }
        
        # Insert into database
        try:
            self.db.execute_query(
                f"""
                INSERT INTO {self.db.tables['correlations']}
                (symbol, start_date, end_date, window_size, price_change, 
                sentiment_change, correlation_value, data_points, calculated_at)
                VALUES
                (%(symbol)s, %(start_date)s, %(end_date)s, %(window_size)s, 
                %(price_change)s, %(sentiment_change)s, %(correlation_value)s, 
                %(data_points)s, %(calculated_at)s)
                """, 
                correlation_data, 
                fetch=False
            )
        except Exception as e:
            logger.error(f"Error storing correlation results: {e}")
    
    def analyze_all_stocks(self, days=7, window_sizes=[30, 60, 240]):
        """
        Analyze correlations for all target stocks.
        
        Args:
            days (int): Number of days to analyze
            window_sizes (list): List of window sizes in minutes to analyze
            
        Returns:
            dict: Dictionary with correlation results for all stocks
        """
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        all_results = {}
        
        for stock in TARGET_STOCKS:
            symbol = stock['symbol']
            stock_results = {}
            
            for window_size in window_sizes:
                logger.info(f"Analyzing correlation for {symbol} with {window_size}min windows")
                correlation = self.calculate_correlation(
                    symbol, 
                    start_date, 
                    end_date, 
                    window_size
                )
                
                stock_results[f'{window_size}min'] = correlation
            
            all_results[symbol] = stock_results
        
        return all_results
    
    def get_top_correlations(self, days=7, window_size=60, top_n=5):
        """
        Get top correlated stocks.
        
        Args:
            days (int): Number of days to analyze
            window_size (int): Window size in minutes
            top_n (int): Number of top correlations to return
            
        Returns:
            list: List of top correlation results
        """
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Fetch stored correlations from database
        query = f"""
        SELECT * FROM {self.db.tables['correlations']}
        WHERE 
            start_date >= %(start_date)s AND
            end_date <= %(end_date)s AND
            window_size = %(window_size)s
        ORDER BY ABS(correlation_value) DESC
        LIMIT %(top_n)s
        """
        
        params = {
            'start_date': start_date,
            'end_date': end_date,
            'window_size': window_size,
            'top_n': top_n
        }
        
        try:
            results = self.db.execute_query(query, params)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error fetching top correlations: {e}")
            return []
    
    def plot_correlation(self, symbol, start_date, end_date, window_size=60):
        """
        Plot correlation between sentiment and price changes.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for analysis
            end_date (datetime): End date for analysis
            window_size (int): Window size in minutes
            
        Returns:
            matplotlib.figure.Figure: Matplotlib figure with correlation plots
        """
        results = self.calculate_correlation(symbol, start_date, end_date, window_size)
        
        if 'aligned_data' not in results or results['aligned_data'] is None:
            logger.warning(f"No aligned data available for plotting {symbol}")
            return None
        
        aligned_data = results['aligned_data']
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)
        
        # Plot 1: Price vs. Time
        ax1.plot(aligned_data['window'], aligned_data['close'], 'b-', label='Close Price')
        ax1.set_ylabel('Price ($)')
        ax1.set_title(f'{symbol} Stock Price')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Sentiment vs. Time
        ax2.plot(aligned_data['window'], aligned_data['compound_score'], 'g-', label='Sentiment')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_ylabel('Sentiment Score')
        ax2.set_title(f'{symbol} Sentiment Scores')
        ax2.legend()
        ax2.grid(True)
        
        # Plot 3: Scatter plot of Price Change vs. Sentiment
        ax3.scatter(aligned_data['compound_score'], aligned_data['price_change'], alpha=0.6)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.axvline(x=0, color='r', linestyle='--')
        
        # Add regression line
        if len(aligned_data) > 1:
            z = np.polyfit(aligned_data['compound_score'], aligned_data['price_change'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(aligned_data['compound_score'].min(), aligned_data['compound_score'].max(), 100)
            ax3.plot(x_range, p(x_range), 'r--')
        
        ax3.set_xlabel('Sentiment Score')
        ax3.set_ylabel('Price Change (%)')
        ax3.set_title(f'Correlation: {results["pearson_correlation"]:.4f} (p-value: {results["pearson_p_value"]:.4f})')
        ax3.grid(True)
        
        # Set common title
        plt.suptitle(f'{symbol} Sentiment vs. Price Correlation Analysis ({window_size}min windows)', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        
        return fig
    
    def generate_correlation_report(self, days=7, window_size=60):
        """
        Generate a correlation analysis report.
        
        Args:
            days (int): Number of days to analyze
            window_size (int): Window size in minutes
            
        Returns:
            str: Markdown-formatted report
        """
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Get correlations for all stocks
        all_results = {}
        for stock in TARGET_STOCKS:
            symbol = stock['symbol']
            correlation = self.calculate_correlation(
                symbol, 
                start_date, 
                end_date, 
                window_size
            )
            all_results[symbol] = correlation
        
        # Sort by absolute correlation
        sorted_results = sorted(
            all_results.values(), 
            key=lambda x: abs(x.get('pearson_correlation', 0)) if x.get('pearson_correlation') is not None else 0,
            reverse=True
        )
        
        # Generate report
        report = [
            f"# Sentiment-Price Correlation Analysis Report",
            f"**Date Range:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
            f"**Window Size:** {window_size} minutes",
            f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Summary of Findings",
            "The following table shows the correlation between sentiment scores and price changes for each stock:",
            "\n| Symbol | Correlation | p-value | Data Points | Avg. Price Change | Avg. Sentiment |",
            "|--------|-------------|---------|-------------|-------------------|----------------|"
        ]
        
        for result in sorted_results:
            if result.get('pearson_correlation') is not None:
                report.append(
                    f"| {result['symbol']} | "
                    f"{result['pearson_correlation']:.4f} | "
                    f"{result['pearson_p_value']:.4f} | "
                    f"{result['data_points']} | "
                    f"{result['price_change_avg']:.2f}% | "
                    f"{result['sentiment_avg']:.4f} |"
                )
        
        report.extend([
            "\n## Interpretation",
            "- **Strong Positive Correlation (> 0.5):** Sentiment and price tend to move in the same direction",
            "- **Strong Negative Correlation (< -0.5):** Sentiment and price tend to move in opposite directions",
            "- **Weak Correlation (-0.5 to 0.5):** Limited relationship between sentiment and price",
            "- **p-value < 0.05:** Correlation is statistically significant",
            "\n## Detailed Analysis",
        ])
        
        for result in sorted_results[:5]:  # Top 5 strongest correlations
            if result.get('pearson_correlation') is not None:
                corr_strength = "strong" if abs(result['pearson_correlation']) > 0.5 else "moderate" if abs(result['pearson_correlation']) > 0.3 else "weak"
                corr_direction = "positive" if result['pearson_correlation'] > 0 else "negative"
                significance = "statistically significant" if result['pearson_p_value'] < 0.05 else "not statistically significant"
                
                report.extend([
                    f"\n### {result['symbol']}",
                    f"- **Correlation:** {result['pearson_correlation']:.4f} ({corr_strength} {corr_direction})",
                    f"- **Statistical Significance:** p-value = {result['pearson_p_value']:.4f} ({significance})",
                    f"- **Data Points:** {result['data_points']}",
                    f"- **Average Price Change:** {result['price_change_avg']:.2f}%",
                    f"- **Average Sentiment Score:** {result['sentiment_avg']:.4f}",
                    f"\n**Interpretation:** The {corr_strength} {corr_direction} correlation indicates that "
                    f"{'sentiment increases tend to be associated with price increases' if result['pearson_correlation'] > 0 else 'sentiment increases tend to be associated with price decreases'}. "
                    f"This correlation is {significance}."
                ])
        
        return "\n".join(report)

if __name__ == "__main__":
    analyzer = CorrelationAnalyzer()
    
    # Example usage
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=7)
    
    # Print correlations for a specific stock
    correlation = analyzer.calculate_correlation('AAPL', start_date, end_date)
    print(f"AAPL Correlation: {correlation['pearson_correlation']:.4f} (p-value: {correlation['pearson_p_value']:.4f})")
    
    # Generate and save plot
    fig = analyzer.plot_correlation('AAPL', start_date, end_date)
    if fig:
        fig.savefig('aapl_correlation.png')
        print("Plot saved as aapl_correlation.png")
    
    # Generate report
    report = analyzer.generate_correlation_report(days=7)
    with open('correlation_report.md', 'w') as f:
        f.write(report)
    print("Report saved as correlation_report.md")