"""
Dashboard template for visualizing stock sentiment and price data.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
from dash.dependencies import State
import logging
import datetime

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_connector import DatabaseConnector
from config.api_config import TARGET_STOCKS
from data_collection.stock_price_predictor import StockPriceCollector
from data_collection.financial_news_collector import FinancialNewsCollector
from data_processing.sentiment_analyzer import SentimentAnalyzer
from analysis.correlation_analyzer import CorrelationAnalyzer
from analysis.time_lag_analyzer import TimeLagAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentDashboard:
    """
    Class for creating and running a Dash dashboard for stock sentiment analysis.
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        self.db = DatabaseConnector()
        self.stock_collector = StockPriceCollector()
        self.news_collector = FinancialNewsCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.time_lag_analyzer = TimeLagAnalyzer()
        
        # Define available stocks
        self.stocks = TARGET_STOCKS
        self.stock_options = [{'label': f"{stock['symbol']} - {stock['name']}", 'value': stock['symbol']} for stock in self.stocks]
        
        # Define time windows
        self.time_windows = [
            {'label': 'Last 4 Hours', 'value': '4h'},
            {'label': 'Last 12 Hours', 'value': '12h'},
            {'label': 'Last 24 Hours', 'value': '24h'},
            {'label': 'Last 3 Days', 'value': '3d'},
            {'label': 'Last 7 Days', 'value': '7d'},
            {'label': 'Last 14 Days', 'value': '14d'},
            {'label': 'Last 30 Days', 'value': '30d'}
        ]
        
        # Initialize the Dash app
        self.app = dash.Dash(
            __name__,
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
            title="Real-time Stock Sentiment Analysis"
        )
        
        # Define app layout
        self._setup_layout()
        
        # Define callbacks
        self._setup_callbacks()
    
    def _setup_layout(self):
        """Set up the dashboard layout."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Real-time Stock Market Sentiment Analysis"),
                html.P("Analyze sentiment from financial news and its correlation with stock price movements"),
            ], className="header"),
            
            # Controls
            html.Div([
                html.Div([
                    html.Label("Select Stock:"),
                    dcc.Dropdown(
                        id="stock-selector",
                        options=self.stock_options,
                        value=self.stock_options[0]['value'],
                        clearable=False
                    )
                ], className="control-element"),
                
                html.Div([
                    html.Label("Time Range:"),
                    dcc.Dropdown(
                        id="time-range-selector",
                        options=self.time_windows,
                        value="24h",
                        clearable=False
                    )
                ], className="control-element"),
                
                html.Div([
                    html.Button("Refresh Data", id="refresh-button", n_clicks=0),
                    html.Div(id="last-update-time")
                ], className="control-element")
            ], className="controls-container"),
            
            # Overview metrics
            html.Div([
                html.Div([
                    html.H3("Current Price"),
                    html.Div(id="current-price", className="metric-value"),
                    html.Div(id="price-change", className="metric-change")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Sentiment Score"),
                    html.Div(id="sentiment-score", className="metric-value"),
                    html.Div(id="sentiment-label", className="metric-label")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Correlation"),
                    html.Div(id="correlation-value", className="metric-value"),
                    html.Div(id="correlation-significance", className="metric-label")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Data Volume"),
                    html.Div(id="data-volume", className="metric-value"),
                    html.Div(id="data-sources", className="metric-label")
                ], className="metric-card")
            ], className="metrics-container"),
            
            # Main charts
            html.Div([
                # Price and sentiment chart
                html.Div([
                    html.H2("Price vs. Sentiment"),
                    dcc.Graph(id="price-sentiment-chart", style={"height": "500px"})
                ], className="chart-container"),
                
                # Sentiment distribution chart
                html.Div([
                    html.H2("Sentiment Distribution"),
                    dcc.Graph(id="sentiment-distribution", style={"height": "500px"})
                ], className="chart-container"),
            ], className="charts-row"),
            
            # Second row of charts
            html.Div([
                # News volume chart
                html.Div([
                    html.H2("News Volume"),
                    dcc.Graph(id="volume-chart", style={"height": "400px"})
                ], className="chart-container"),
                
                # Lag correlation chart
                html.Div([
                    html.H2("Time Lag Analysis"),
                    dcc.Graph(id="lag-correlation-chart", style={"height": "400px"})
                ], className="chart-container"),
            ], className="charts-row"),
            
            # News content
            html.Div([
                html.H2("Recent News Headlines"),
                html.Div(id="news-table", className="content-table")
            ], className="content-container"),
            
            # Hidden divs for storing data
            html.Div(id="store-price-data", style={"display": "none"}),
            html.Div(id="store-sentiment-data", style={"display": "none"}),
            html.Div(id="store-news-data", style={"display": "none"}),
            
            # Footer
            html.Div([
                html.P("Stock Market Sentiment Analysis Dashboard"),
                html.P("Data sources: Financial news APIs, Stock Price APIs"),
                html.P(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            ], className="footer")
        ], className="dashboard-container")
    
    def _setup_callbacks(self):
        """Set up the dashboard callbacks."""
        
        @self.app.callback(
            [
                Output("store-price-data", "children"),
                Output("store-sentiment-data", "children"),
                Output("store-news-data", "children"),
                Output("last-update-time", "children")
            ],
            [
                Input("stock-selector", "value"),
                Input("time-range-selector", "value"),
                Input("refresh-button", "n_clicks")
            ]
        )
        def update_data_stores(symbol, time_range, n_clicks):
            """Update the data stores with fresh data."""
            # Convert time range to timedelta
            end_date = datetime.datetime.now()
            if time_range.endswith('h'):
                hours = int(time_range[:-1])
                start_date = end_date - datetime.timedelta(hours=hours)
            elif time_range.endswith('d'):
                days = int(time_range[:-1])
                start_date = end_date - datetime.timedelta(days=days)
            else:
                # Default to 1 day
                start_date = end_date - datetime.timedelta(days=1)
            
            # Fetch price data
            price_data = self.db.get_stock_prices(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                limit=10000
            )
            
            # Fetch sentiment data
            sentiment_data = self.db.get_sentiment_scores(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                limit=10000
            )
            
            # Fetch news data
            news_data = self.db.get_news_articles(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                limit=100
            )
            
            # Convert to DataFrames and then to JSON for storage
            price_df = pd.DataFrame(price_data) if price_data else pd.DataFrame()
            sentiment_df = pd.DataFrame(sentiment_data) if sentiment_data else pd.DataFrame()
            news_df = pd.DataFrame(news_data) if news_data else pd.DataFrame()
            
            update_time = f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return (
                price_df.to_json(date_format='iso', orient='split') if not price_df.empty else None,
                sentiment_df.to_json(date_format='iso', orient='split') if not sentiment_df.empty else None,
                news_df.to_json(date_format='iso', orient='split') if not news_df.empty else None,
                update_time
            )
        
        @self.app.callback(
            [
                Output("current-price", "children"),
                Output("price-change", "children"),
                Output("price-change", "className")
            ],
            [Input("store-price-data", "children")]
        )
        def update_price_metrics(price_data_json):
            """Update the price metrics."""
            if not price_data_json:
                return "$0.00", "0.00%", "metric-change neutral"
            
            # Parse the price data
            price_df = pd.read_json(price_data_json, orient='split')
            
            if price_df.empty:
                return "$0.00", "0.00%", "metric-change neutral"
            
            # Sort by timestamp and get latest price
            price_df = price_df.sort_values('timestamp')
            latest_price = price_df['close'].iloc[-1]
            
            # Calculate price change
            first_price = price_df['open'].iloc[0]
            price_change_pct = (latest_price - first_price) / first_price * 100
            
            # Format for display
            price_display = f"${latest_price:.2f}"
            change_display = f"{price_change_pct:+.2f}%"
            change_class = "metric-change positive" if price_change_pct >= 0 else "metric-change negative"
            
            return price_display, change_display, change_class
        
        @self.app.callback(
            [
                Output("sentiment-score", "children"),
                Output("sentiment-label", "children"),
                Output("sentiment-score", "className"),
                Output("sentiment-label", "className")
            ],
            [Input("store-sentiment-data", "children")]
        )
        def update_sentiment_metrics(sentiment_data_json):
            """Update the sentiment metrics."""
            if not sentiment_data_json:
                return "0.00", "Neutral", "metric-value neutral", "metric-label neutral"
            
            # Parse the sentiment data
            sentiment_df = pd.read_json(sentiment_data_json, orient='split')
            
            if sentiment_df.empty:
                return "0.00", "Neutral", "metric-value neutral", "metric-label neutral"
            
            # Calculate average sentiment
            avg_sentiment = sentiment_df['compound_score'].mean()
            
            # Determine sentiment label
            if avg_sentiment >= 0.05:
                label = "Positive"
                class_name = "positive"
            elif avg_sentiment <= -0.05:
                label = "Negative"
                class_name = "negative"
            else:
                label = "Neutral"
                class_name = "neutral"
            
            # Format for display
            sentiment_display = f"{avg_sentiment:.2f}"
            
            return sentiment_display, label, f"metric-value {class_name}", f"metric-label {class_name}"
        
        @self.app.callback(
            [
                Output("correlation-value", "children"),
                Output("correlation-significance", "children"),
                Output("correlation-value", "className")
            ],
            [
                Input("store-price-data", "children"),
                Input("store-sentiment-data", "children")
            ]
        )
        def update_correlation_metrics(price_data_json, sentiment_data_json):
            """Update the correlation metrics."""
            if not price_data_json or not sentiment_data_json:
                return "0.00", "Not significant", "metric-value neutral"
            
            # Parse the data
            price_df = pd.read_json(price_data_json, orient='split')
            sentiment_df = pd.read_json(sentiment_data_json, orient='split')
            
            if price_df.empty or sentiment_df.empty:
                return "0.00", "Not significant", "metric-value neutral"
            
            try:
                # Calculate price changes
                price_df = price_df.sort_values('timestamp')
                price_df['price_change'] = price_df['close'].pct_change() * 100
                
                # Create a common time resolution (hourly)
                price_df['hour'] = pd.to_datetime(price_df['timestamp']).dt.floor('H')
                sentiment_df['hour'] = pd.to_datetime(sentiment_df['timestamp']).dt.floor('H')
                
                # Aggregate by hour
                price_hourly = price_df.groupby('hour')['price_change'].mean().reset_index()
                sentiment_hourly = sentiment_df.groupby('hour')['compound_score'].mean().reset_index()
                
                # Merge on hour
                merged = pd.merge(price_hourly, sentiment_hourly, on='hour', how='inner')
                
                if len(merged) < 3:
                    return "Insufficient data", "Need more data points", "metric-value neutral"
                
                # Calculate correlation
                from scipy.stats import pearsonr
                correlation, p_value = pearsonr(merged['compound_score'], merged['price_change'])
                
                # Format for display
                correlation_display = f"{correlation:.2f}"
                
                if p_value < 0.05:
                    significance = "Significant (p<0.05)"
                    if correlation > 0:
                        class_name = "positive"
                    else:
                        class_name = "negative"
                else:
                    significance = f"Not significant (p={p_value:.2f})"
                    class_name = "neutral"
                
                return correlation_display, significance, f"metric-value {class_name}"
                
            except Exception as e:
                logger.error(f"Error calculating correlation: {e}")
                return "Error", "Calculation failed", "metric-value neutral"
        
        @self.app.callback(
            [
                Output("data-volume", "children"),
                Output("data-sources", "children")
            ],
            [
                Input("store-sentiment-data", "children"),
                Input("store-news-data", "children")
            ]
        )
        def update_data_volume_metrics(sentiment_data_json, news_data_json):
            """Update the data volume metrics."""
            sentiment_count = 0
            news_count = 0
            
            if sentiment_data_json:
                sentiment_df = pd.read_json(sentiment_data_json, orient='split')
                sentiment_count = len(sentiment_df)
            
            if news_data_json:
                news_df = pd.read_json(news_data_json, orient='split')
                news_count = len(news_df)
            
            # Format for display
            volume_display = f"{sentiment_count:,}"
            sources_display = f"News articles: {news_count}"
            
            return volume_display, sources_display
        
        @self.app.callback(
            Output("price-sentiment-chart", "figure"),
            [
                Input("store-price-data", "children"),
                Input("store-sentiment-data", "children")
            ]
        )
        def update_price_sentiment_chart(price_data_json, sentiment_data_json):
            """Update the price and sentiment chart."""
            if not price_data_json or not sentiment_data_json:
                # Return empty figure
                return go.Figure()
            
            # Parse the data
            price_df = pd.read_json(price_data_json, orient='split')
            sentiment_df = pd.read_json(sentiment_data_json, orient='split')
            
            if price_df.empty or sentiment_df.empty:
                return go.Figure()
            
            # Process the data
            price_df = price_df.sort_values('timestamp')
            
            # Create subplots with shared x-axis
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               vertical_spacing=0.05,
                               row_heights=[0.7, 0.3])
            
            # Add price candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=price_df['timestamp'],
                    open=price_df['open'],
                    high=price_df['high'],
                    low=price_df['low'],
                    close=price_df['close'],
                    name="Price"
                ),
                row=1, col=1
            )
            
            # Add sentiment scatter plot
            fig.add_trace(
                go.Scatter(
                    x=sentiment_df['timestamp'],
                    y=sentiment_df['compound_score'],
                    mode='markers+lines',
                    name="Sentiment",
                    marker=dict(
                        color=sentiment_df['compound_score'],
                        colorscale='RdYlGn',
                        cmin=-0.5,
                        cmax=0.5,
                        size=5,
                        showscale=True
                    )
                ),
                row=2, col=1
            )
            
            # Add horizontal line at y=0 for sentiment
            fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", row=2, col=1)
            
            # Update layout
            fig.update_layout(
                title=f"Price vs. Sentiment",
                height=600,
                xaxis_rangeslider_visible=False,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            # Update yaxis for sentiment
            fig.update_yaxes(title_text="Sentiment Score", range=[-1, 1], row=2, col=1)
            
            # Update yaxis for price
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            
            return fig
        
        @self.app.callback(
            Output("sentiment-distribution", "figure"),
            [Input("store-sentiment-data", "children")]
        )
        def update_sentiment_distribution(sentiment_data_json):
            """Update the sentiment distribution chart."""
            if not sentiment_data_json:
                return go.Figure()
            
            # Parse the sentiment data
            sentiment_df = pd.read_json(sentiment_data_json, orient='split')
            
            if sentiment_df.empty:
                return go.Figure()
            
            # Create subplots for different views
            fig = make_subplots(rows=1, cols=2, 
                               subplot_titles=("Sentiment Distribution", "Sentiment by Source"),
                               specs=[[{"type": "histogram"}, {"type": "bar"}]])
            
            # Add histogram of sentiment scores
            fig.add_trace(
                go.Histogram(
                    x=sentiment_df['compound_score'],
                    nbinsx=20,
                    marker_color=sentiment_df['compound_score'],
                    marker_colorscale='RdYlGn',
                    marker_cmin=-1,
                    marker_cmax=1,
                    name="Distribution"
                ),
                row=1, col=1
            )
            
            # Add sentiment by source
            if 'source' in sentiment_df.columns:
                source_sentiment = sentiment_df.groupby('source')['compound_score'].agg(['mean', 'count']).reset_index()
                source_sentiment = source_sentiment.sort_values('mean', ascending=False)
                
                fig.add_trace(
                    go.Bar(
                        x=source_sentiment['source'],
                        y=source_sentiment['mean'],
                        marker_color=source_sentiment['mean'],
                        marker_colorscale='RdYlGn',
                        marker_cmin=-0.5,
                        marker_cmax=0.5,
                        text=source_sentiment['count'],
                        name="By Source"
                    ),
                    row=1, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="Sentiment Analysis",
                height=500,
                bargap=0.01,
                margin=dict(l=50, r=50, t=80, b=50),
                showlegend=False
            )
            
            # Update axes
            fig.update_xaxes(title_text="Sentiment Score", row=1, col=1)
            fig.update_yaxes(title_text="Count", row=1, col=1)
            fig.update_xaxes(title_text="Source", row=1, col=2)
            fig.update_yaxes(title_text="Avg. Sentiment", range=[-1, 1], row=1, col=2)
            
            return fig
        
        @self.app.callback(
            Output("volume-chart", "figure"),
            [
                Input("store-news-data", "children")
            ]
        )
        def update_volume_chart(news_data_json):
            """Update the news volume chart."""
            if not news_data_json:
                return go.Figure()
            
            # Initialize an empty dataframe for the timeline
            timeline_df = pd.DataFrame()
            
            if news_data_json:
                news_df = pd.read_json(news_data_json, orient='split')
                if not news_df.empty and 'published_at' in news_df.columns:
                    # Create hourly bins for news
                    news_df['hour'] = pd.to_datetime(news_df['published_at']).dt.floor('H')
                    news_hourly = news_df.groupby('hour').size().reset_index(name='news_count')
                    
                    if timeline_df.empty:
                        timeline_df = news_hourly
                    else:
                        timeline_df = pd.merge(timeline_df, news_hourly, on='hour', how='outer')
            
            # Fill NaN values with 0
            timeline_df = timeline_df.fillna(0)
            
            if timeline_df.empty:
                return go.Figure()
            
            # Sort by time
            timeline_df = timeline_df.sort_values('hour')
            
            # Create figure
            fig = go.Figure()
            
            # Add bars for news count
            if 'news_count' in timeline_df.columns:
                fig.add_trace(
                    go.Bar(
                        x=timeline_df['hour'],
                        y=timeline_df['news_count'],
                        name="News Articles",
                        marker_color='rgba(58, 71, 80, 0.6)'
                    )
                )
            
            # Update layout
            fig.update_layout(
                title="News Volume Over Time",
                xaxis_title="Time",
                yaxis_title="Count",
                height=400,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            
            return fig
        
        @self.app.callback(
            Output("lag-correlation-chart", "figure"),
            [
                Input("store-price-data", "children"),
                Input("store-sentiment-data", "children"),
                Input("stock-selector", "value")
            ]
        )
        def update_lag_correlation_chart(price_data_json, sentiment_data_json, symbol):
            """Update the lag correlation chart."""
            if not price_data_json or not sentiment_data_json:
                return go.Figure()
            
            try:
                # Parse the data
                price_df = pd.read_json(price_data_json, orient='split')
                sentiment_df = pd.read_json(sentiment_data_json, orient='split')
                
                if price_df.empty or sentiment_df.empty or len(price_df) < 10:
                    # Not enough data for lag analysis
                    fig = go.Figure()
                    fig.add_annotation(
                        x=0.5, y=0.5,
                        text="Not enough data for lag analysis",
                        showarrow=False,
                        font=dict(size=16)
                    )
                    return fig
                
                # Convert timestamps to datetime
                price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
                sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
                
                # Get the date range
                start_date = min(price_df['timestamp'].min(), sentiment_df['timestamp'].min())
                end_date = max(price_df['timestamp'].max(), sentiment_df['timestamp'].max())
                
                # Run lag analysis
                lag_results = self.time_lag_analyzer.analyze_lag_correlation(
                    symbol, start_date, end_date, window_size=60, max_lag=10
                )
                
                if not lag_results or 'lag_correlations' not in lag_results or not lag_results['lag_correlations']:
                    # Lag analysis failed
                    fig = go.Figure()
                    fig.add_annotation(
                        x=0.5, y=0.5,
                        text="Lag analysis failed or returned no results",
                        showarrow=False,
                        font=dict(size=16)
                    )
                    return fig
                
                # Extract data for plotting
                lags = [lc['lag'] for lc in lag_results['lag_correlations']]
                correlations = [lc['correlation'] for lc in lag_results['lag_correlations']]
                
                # Create figure
                fig = go.Figure()
                
                # Add bar chart for correlations
                fig.add_trace(
                    go.Bar(
                        x=lags,
                        y=correlations,
                        marker_color='rgba(58, 71, 80, 0.6)',
                        name="Correlation"
                    )
                )
                
                # Highlight best lag
                if 'best_lag' in lag_results and lag_results['best_lag'] in lags:
                    best_idx = lags.index(lag_results['best_lag'])
                    fig.add_trace(
                        go.Bar(
                            x=[lags[best_idx]],
                            y=[correlations[best_idx]],
                            marker_color='rgba(246, 78, 139, 0.8)',
                            name=f"Best Lag: {lag_results['best_lag']} periods"
                        )
                    )
                
                # Add reference lines
                fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray")
                fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="gray")
                
                # Update layout
                fig.update_layout(
                    title=f"Sentiment-Price Lag Analysis",
                    xaxis_title="Lag (Time Periods)",
                    yaxis_title="Correlation",
                    height=400,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating lag correlation chart: {e}")
                fig = go.Figure()
                fig.add_annotation(
                    x=0.5, y=0.5,
                    text=f"Error: {str(e)}",
                    showarrow=False,
                    font=dict(size=16)
                )
                return fig
        
        @self.app.callback(
            Output("news-table", "children"),
            [Input("store-news-data", "children")]
        )
        def update_news_table(news_data_json):
            """Update the news headlines table."""
            if not news_data_json:
                return html.P("No news data available.")
            
            news_df = pd.read_json(news_data_json, orient='split')
            
            if news_df.empty:
                return html.P("No news data available.")
            
            # Sort by published date (newest first)
            if 'published_at' in news_df.columns:
                news_df = news_df.sort_values('published_at', ascending=False)
            
            # Create news cards
            news_cards = []
            for i, row in news_df.iterrows():
                if i >= 10:  # Limit to 10 news items
                    break
                
                # Get sentiment score if available
                sentiment_label = ""
                if 'sentiment_label' in row:
                    sentiment_label = f" ({row['sentiment_label']})"
                elif 'compound_score' in row:
                    if row['compound_score'] >= 0.05:
                        sentiment_label = " (Positive)"
                    elif row['compound_score'] <= -0.05:
                        sentiment_label = " (Negative)"
                    else:
                        sentiment_label = " (Neutral)"
                
                # Format date
                date_str = ""
                if 'published_at' in row:
                    date = pd.to_datetime(row['published_at'])
                    date_str = date.strftime("%Y-%m-%d %H:%M")
                
                # Create card
                card = html.Div([
                    html.H4(row['headline']),
                    html.P(row['summary'] if 'summary' in row and pd.notna(row['summary']) else ""),
                    html.Div([
                        html.Span(f"Source: {row['source']}"),
                        html.Span(f" | {date_str}"),
                        html.Span(sentiment_label, className=sentiment_label.lower().strip('() '))
                    ], className="news-metadata"),
                    html.A("Read More", href=row['url'] if 'url' in row else "#", target="_blank")
                ], className="news-card")
                
                news_cards.append(card)
            
            return html.Div(news_cards)
    
    def run_server(self, debug=True, port=8050):
        """Run the dashboard server."""
        # Use app.run() instead of app.run_server() for newer versions of Dash
        self.app.run(debug=debug, port=port)

if __name__ == "__main__":
    dashboard = SentimentDashboard()
    dashboard.run_server()