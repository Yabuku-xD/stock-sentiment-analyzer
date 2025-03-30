"""
Main module for Real-time Stock Market Sentiment Analysis.
Coordinates data collection, processing, analysis and visualization.
"""

import os
import sys
import time
import argparse
import threading
import logging
import datetime
import pandas as pd
from pathlib import Path

# Create necessary directories before configuring logging
directories = [
    'logs',
    'data',
    'models',
    'reports',
    'reports/analysis',
    'reports/models',
    'visualizations'
]

for directory in directories:
    Path(directory).mkdir(exist_ok=True, parents=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
from config.api_config import TARGET_STOCKS
from data_collection.stock_price_predictor import StockPriceCollector
from data_collection.financial_news_collector import FinancialNewsCollector
from data_processing.sentiment_analyzer import SentimentAnalyzer
from data_processing.entity_extraction import EntityExtractor
from database.db_connector import DatabaseConnector
from analysis.correlation_analyzer import CorrelationAnalyzer
from analysis.time_lag_analyzer import TimeLagAnalyzer
from analysis.predictive_model import PredictiveModel
from visualization.dashboard_template import SentimentDashboard

class StockSentimentAnalysis:
    """
    Main class for coordinating the stock sentiment analysis pipeline.
    """
    
    def __init__(self):
        """Initialize the stock sentiment analysis system."""
        # Create data directories if they don't exist
        self._create_directories()
        
        # Initialize database first
        self.db = DatabaseConnector()
        
        # Then initialize components after database is ready
        self.stock_collector = StockPriceCollector()
        self.news_collector = FinancialNewsCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.entity_extractor = EntityExtractor()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.time_lag_analyzer = TimeLagAnalyzer()
        self.predictive_model = PredictiveModel()
        self.dashboard = SentimentDashboard()
        
        # Collection threads
        self.collection_threads = {}
        self.stop_event = threading.Event()
    
    def _create_directories(self):
        """Create necessary directories for the project."""
        directories = [
            'logs',
            'data',
            'models',
            'reports',
            'reports/analysis',
            'reports/models',
            'visualizations'
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True, parents=True)
    
    def collect_data(self, continuous=False):
        """
        Collect data from various sources.
        
        Args:
            continuous (bool): Whether to collect data continuously
        """
        if continuous:
            # Start continuous collection in separate threads
            self._start_continuous_collection()
        else:
            # Collect data once
            logger.info("Starting one-time data collection")
            
            # Collect stock prices
            stock_data = self.stock_collector.collect_latest_prices()
            logger.info(f"Collected {len(stock_data)} stock price records")
            
            # Collect news
            news_data = self.news_collector.collect_latest_news()
            logger.info(f"Collected {len(news_data)} news articles")
            
            return {
                'stock_data': stock_data,
                'news_data': news_data
            }
    
    def _start_continuous_collection(self):
        """Start continuous data collection in separate threads."""
        logger.info("Starting continuous data collection")
        
        # Stock price collection thread
        stock_thread = threading.Thread(
            target=self._continuous_collection_worker,
            args=(self.stock_collector.run_collection, "stock_prices"),
            daemon=True
        )
        self.collection_threads['stock_prices'] = stock_thread
        logger.info("Starting stock_prices collection thread")
        stock_thread.start()
        
        # News collection thread
        news_thread = threading.Thread(
            target=self._continuous_collection_worker,
            args=(self.news_collector.run_collection, "news"),
            daemon=True
        )
        self.collection_threads['news'] = news_thread
        logger.info("Starting news collection thread")
        news_thread.start()
    
    def _continuous_collection_worker(self, collection_func, name):
        """
        Worker function for continuous data collection.
        
        Args:
            collection_func (function): Data collection function to run
            name (str): Name of the collection process
        """
        try:
            collection_func()
        except Exception as e:
            logger.error(f"Error in {name} collection thread: {e}")
    
    def stop_collection(self):
        """Stop continuous data collection."""
        logger.info("Stopping data collection")
        self.stop_event.set()
        
        # Wait for threads to finish
        for name, thread in self.collection_threads.items():
            logger.info(f"Waiting for {name} thread to finish")
            thread.join(timeout=5)
        
        self.collection_threads = {}
        self.stop_event.clear()
    
    def process_data(self, days=1):
        """
        Process collected data: sentiment analysis and entity extraction.
        
        Args:
            days (int): Number of days of data to process
            
        Returns:
            dict: Dictionary with processing results
        """
        logger.info(f"Processing data from the last {days} days")
        
        # Define the time range
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Process news articles
        logger.info("Processing news articles")
        news_articles = self.db.get_news_articles(start_date=start_date)
        
        if not news_articles:
            logger.warning("No news articles found for processing")
            return {
                'news_sentiment': [],
                'news_entities': []
            }
        
        news_sentiment = self.sentiment_analyzer.analyze_news_batch(news_articles=news_articles)
        news_entities = self.entity_extractor.process_news_batch(news_articles=news_articles)
        
        return {
            'news_sentiment': news_sentiment,
            'news_entities': news_entities
        }
    
    def analyze_data(self, days=7, window_sizes=[30, 60, 240]):
        """
        Analyze processed data: correlations and time lag analysis.
        
        Args:
            days (int): Number of days of data to analyze
            window_sizes (list): List of window sizes in minutes to analyze
            
        Returns:
            dict: Dictionary with analysis results
        """
        logger.info(f"Analyzing data from the last {days} days")
        
        # Define the time range
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Analyze correlations
        logger.info("Analyzing correlations")
        correlations = self.correlation_analyzer.analyze_all_stocks(days, window_sizes)
        
        # Analyze time lags
        logger.info("Analyzing time lags")
        time_lags = self.time_lag_analyzer.analyze_all_stocks(days, window_sizes[1])
        
        # For each stock, get the sentiment summary
        sentiment_summaries = {}
        for stock in TARGET_STOCKS:
            symbol = stock['symbol']
            sentiment_summaries[symbol] = self.sentiment_analyzer.get_sentiment_summary(symbol, days)
        
        return {
            'correlations': correlations,
            'time_lags': time_lags,
            'sentiment_summaries': sentiment_summaries
        }
    
    def build_models(self, days=30, model_type='random_forest'):
        """
        Build predictive models for stock price movements.
        
        Args:
            days (int): Number of days of data to use for training
            model_type (str): Type of model to build
            
        Returns:
            dict: Dictionary with model results
        """
        logger.info(f"Building {model_type} models using {days} days of data")
        
        # Build models for all stocks
        model_results = self.predictive_model.build_models_for_all_stocks(days, 60, model_type)
        
        # Generate model reports
        report_dir = Path('reports/models')
        report_dir.mkdir(exist_ok=True, parents=True)
        
        for symbol, result in model_results.items():
            if result['success']:
                # Generate report
                report = self.predictive_model.generate_model_report(result)
                report_path = report_dir / f"{symbol}_{model_type}_model_report.md"
                
                with open(report_path, 'w') as f:
                    f.write(report)
                
                logger.info(f"Saved model report for {symbol} to {report_path}")
                
                # Generate and save plots
                fig1 = self.predictive_model.plot_feature_importance(result)
                if fig1:
                    plot_path = report_dir / f"{symbol}_{model_type}_feature_importance.png"
                    fig1.savefig(plot_path)
                
                fig2 = self.predictive_model.plot_confusion_matrix(result)
                if fig2:
                    plot_path = report_dir / f"{symbol}_{model_type}_confusion_matrix.png"
                    fig2.savefig(plot_path)
        
        return model_results
    
    def generate_reports(self, days=7):
        """
        Generate analysis reports.
        
        Args:
            days (int): Number of days of data to include in reports
            
        Returns:
            dict: Dictionary with report paths
        """
        logger.info(f"Generating reports for the last {days} days")
        
        # Define the time range
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Create reports directory
        report_dir = Path('reports/analysis')
        report_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate correlation report
        correlation_report = self.correlation_analyzer.generate_correlation_report(days)
        correlation_report_path = report_dir / f"correlation_report_{end_date.strftime('%Y%m%d')}.md"
        
        with open(correlation_report_path, 'w') as f:
            f.write(correlation_report)
        
        logger.info(f"Saved correlation report to {correlation_report_path}")
        
        # Generate time lag report
        lag_report = self.time_lag_analyzer.generate_lag_report(days)
        lag_report_path = report_dir / f"time_lag_report_{end_date.strftime('%Y%m%d')}.md"
        
        with open(lag_report_path, 'w') as f:
            f.write(lag_report)
        
        logger.info(f"Saved time lag report to {lag_report_path}")
        
        # Generate plots for a few stocks
        plot_dir = Path('visualizations')
        plot_dir.mkdir(exist_ok=True)
        
        plot_paths = {}
        for stock in TARGET_STOCKS[:5]:  # First 5 stocks
            symbol = stock['symbol']
            
            # Correlation plot
            corr_fig = self.correlation_analyzer.plot_correlation(symbol, start_date, end_date)
            if corr_fig:
                corr_path = plot_dir / f"{symbol}_correlation_{end_date.strftime('%Y%m%d')}.png"
                corr_fig.savefig(corr_path)
                plot_paths[f"{symbol}_correlation"] = str(corr_path)
            
            # Lag analysis plot
            lag_fig = self.time_lag_analyzer.plot_lag_analysis(symbol, start_date, end_date)
            if lag_fig:
                lag_path = plot_dir / f"{symbol}_lag_analysis_{end_date.strftime('%Y%m%d')}.png"
                lag_fig.savefig(lag_path)
                plot_paths[f"{symbol}_lag_analysis"] = str(lag_path)
        
        return {
            'correlation_report': str(correlation_report_path),
            'lag_report': str(lag_report_path),
            'plots': plot_paths
        }
    
    def run_dashboard(self, debug=True, port=8050):
        """
        Run the interactive dashboard.
        
        Args:
            debug (bool): Whether to run in debug mode
            port (int): Port number to run the dashboard on
        """
        logger.info(f"Starting dashboard on port {port}")
        self.dashboard.run_server(debug=debug, port=port)
    
    def run_pipeline(self, collect=True, process=True, analyze=True, model=True, report=True, dashboard=True):
        """
        Run the complete analysis pipeline.
        
        Args:
            collect (bool): Whether to collect data
            process (bool): Whether to process data
            analyze (bool): Whether to analyze data
            model (bool): Whether to build predictive models
            report (bool): Whether to generate reports
            dashboard (bool): Whether to run the dashboard
        """
        logger.info("Starting the complete analysis pipeline")
        
        # Start continuous data collection if needed
        if collect:
            self.collect_data(continuous=True)
        
        try:
            # Process data
            if process:
                self.process_data(days=1)
            
            # Analyze data
            if analyze:
                self.analyze_data(days=7)
            
            # Build models
            if model:
                self.build_models(days=30)
            
            # Generate reports
            if report:
                self.generate_reports(days=7)
            
            # Run dashboard
            if dashboard:
                self.run_dashboard()
                
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted")
        finally:
            # Stop data collection
            if collect:
                self.stop_collection()
            
            logger.info("Pipeline completed")

def main():
    """
    Main function to run the stock sentiment analysis system.
    
    Example usage:
    - To run the complete pipeline: python main.py --all
    - To run only data collection: python main.py --collect
    - To run dashboard with existing data: python main.py --dashboard
    """
    parser = argparse.ArgumentParser(description='Real-time Stock Market Sentiment Analysis')
    
    parser.add_argument('--collect', action='store_true', help='Collect data from sources')
    parser.add_argument('--process', action='store_true', help='Process collected data')
    parser.add_argument('--analyze', action='store_true', help='Analyze processed data')
    parser.add_argument('--model', action='store_true', help='Build predictive models')
    parser.add_argument('--report', action='store_true', help='Generate analysis reports')
    parser.add_argument('--dashboard', action='store_true', help='Run the dashboard')
    parser.add_argument('--all', action='store_true', help='Run the complete pipeline')
    parser.add_argument('--initialize-db', action='store_true', help='Initialize database tables')
    
    parser.add_argument('--days', type=int, default=7, help='Number of days of data to use')
    parser.add_argument('--port', type=int, default=8050, help='Port number for dashboard')
    parser.add_argument('--model-type', type=str, default='random_forest', 
                        choices=['random_forest', 'logistic_regression', 'svm', 'xgboost'],
                        help='Type of predictive model to build')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any([args.collect, args.process, args.analyze, args.model, 
                args.report, args.dashboard, args.all, args.initialize_db]):
        parser.print_help()
        return
    
    # Create the analysis system
    system = StockSentimentAnalysis()
    
    # Just initialize database and exit if requested
    if args.initialize_db:
        system.db.initialize_database()
        print("Database initialized successfully")
        return
    
    # Run the requested components
    if args.all:
        system.run_pipeline()
    else:
        if args.collect:
            system.collect_data(continuous=True)
        
        if args.process:
            system.process_data(days=args.days)
        
        if args.analyze:
            system.analyze_data(days=args.days)
        
        if args.model:
            system.build_models(days=args.days, model_type=args.model_type)
        
        if args.report:
            system.generate_reports(days=args.days)
        
        if args.dashboard:
            system.run_dashboard(port=args.port)
        
        # If we started data collection, stop it before exiting
        if args.collect:
            try:
                # Keep the main thread alive if running only collection
                if not any([args.process, args.analyze, args.model, args.report, args.dashboard]):
                    print("Press Ctrl+C to stop data collection and exit")
                    while True:
                        time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                system.stop_collection()

if __name__ == "__main__":
    main()