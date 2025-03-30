# Real-time Stock Market Sentiment Analysis & Correlation

This project analyzes real-time financial news headlines and social media sentiment related to specific stocks or sectors, correlating this sentiment with stock price movements.

## Features

- Real-time collection of financial news and stock price data
- NLP-based sentiment analysis using FinBERT and VADER
- Entity extraction for stock tickers and company names
- Time-series database for storing and querying financial and sentiment data
- Correlation analysis between sentiment and price movements
- Interactive dashboard for visualization

## Setup and Installation

1. Clone this repository
2. Install required packages: `pip install -r requirements.txt`
3. Set up database using the SQL scripts in the `database` folder
4. Configure API keys in `config/api_config.py`
5. Run the main application: `python main.py`

## Data Sources

- Financial News: Finnhub, NewsAPI
- Stock Prices: Alpha Vantage, Yahoo Finance
- Social Media: Reddit (r/wallstreetbets), StockTwits

## Dashboard

The dashboard displays real-time stock price charts alongside sentiment trends, allowing you to:
- View sentiment distribution for selected stocks/sectors
- Analyze correlations between sentiment and price/volume changes
- Filter by stock ticker, time range, and data source

## License

This project is licensed under the MIT License - see the LICENSE file for details.