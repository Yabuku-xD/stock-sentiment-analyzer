"""
Configuration file for API keys and endpoints.
Replace placeholder values with your actual API keys.
"""

# Financial News API configurations
FINNHUB_API_KEY = "cvkoic9r01qtnb8tv5mgcvkoic9r01qtnb8tv5n0"
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"

NEWS_API_KEY = "d461bca7b26e485abd731a2acbab10e0"
NEWS_API_BASE_URL = "https://newsapi.org/v2"

# Target stocks to monitor (examples)
TARGET_STOCKS = [
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN", "name": "Amazon.com, Inc."},
    {"symbol": "TSLA", "name": "Tesla, Inc."},
    {"symbol": "META", "name": "Meta Platforms, Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
    {"symbol": "V", "name": "Visa Inc."},
    {"symbol": "WMT", "name": "Walmart Inc."}
]

# News update frequency in seconds
NEWS_UPDATE_INTERVAL = 300  # 5 minutes

# Stock price update frequency in seconds
PRICE_UPDATE_INTERVAL = 60  # 1 minute