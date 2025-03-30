"""
Sentiment analysis module for financial text data.
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import datetime

# Add parent directory to path to import from project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.text_preprocessor import TextPreprocessor
from database.db_connector import DatabaseConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Class for analyzing sentiment in financial texts using multiple models.
    """
    
    def __init__(self, use_finbert=True, use_vader=True):
        """
        Initialize the sentiment analyzer with chosen models.
        
        Args:
            use_finbert (bool): Whether to use FinBERT model
            use_vader (bool): Whether to use VADER model
        """
        self.preprocessor = TextPreprocessor()
        self.db = DatabaseConnector()
        
        self.use_finbert = use_finbert
        self.use_vader = use_vader
        
        # Initialize VADER
        if use_vader:
            try:
                import nltk
                try:
                    nltk.data.find('sentiment/vader_lexicon.zip')
                except LookupError:
                    nltk.download('vader_lexicon')
                self.vader = SentimentIntensityAnalyzer()
                logger.info("VADER sentiment analyzer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize VADER: {e}")
                self.use_vader = False
        
        # Initialize FinBERT
        if use_finbert:
            try:
                # Load pre-trained FinBERT model
                self.finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                self.finbert_model.eval()  # Set to evaluation mode
                logger.info("FinBERT model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize FinBERT: {e}")
                self.use_finbert = False
        
        # Financial terms with positive/negative sentiment
        self.financial_pos_terms = {
            'beat', 'exceeded', 'surpassed', 'outperform', 'strong', 'growth',
            'profit', 'gain', 'positive', 'upgrade', 'bullish', 'upside',
            'record', 'buy', 'opportunity', 'rally', 'recover', 'success'
        }
        
        self.financial_neg_terms = {
            'miss', 'missed', 'below', 'weak', 'decline', 'drop', 'loss',
            'negative', 'downgrade', 'bearish', 'downside', 'sell', 'risk',
            'concern', 'warning', 'disappoint', 'fail', 'underperform'
        }
    
    def get_vader_sentiment(self, text):
        """
        Get sentiment score using VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: VADER sentiment scores
        """
        if not self.use_vader:
            return {
                'compound': 0.0, 
                'positive': 0.0, 
                'neutral': 0.0, 
                'negative': 0.0
            }
        
        scores = self.vader.polarity_scores(text)
        return scores
    
    def get_finbert_sentiment(self, text):
        """
        Get sentiment score using FinBERT.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: FinBERT sentiment scores (positive, negative, neutral)
        """
        if not self.use_finbert:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        # Truncate text if too long for FinBERT
        max_length = self.finbert_tokenizer.model_max_length
        if len(text.split()) > max_length:
            text = ' '.join(text.split()[:max_length])
        
        inputs = self.finbert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.finbert_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # FinBERT classes: positive (0), negative (1), neutral (2)
        scores = {
            'positive': predictions[0][0].item(),
            'negative': predictions[0][1].item(),
            'neutral': predictions[0][2].item()
        }
        
        return scores
    
    def get_rule_based_sentiment(self, text):
        """
        Get sentiment using rule-based approach with financial terms.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Rule-based sentiment scores
        """
        words = text.lower().split()
        
        pos_count = sum(1 for word in words if word in self.financial_pos_terms)
        neg_count = sum(1 for word in words if word in self.financial_neg_terms)
        
        total = pos_count + neg_count
        if total == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        positive = pos_count / len(words)
        negative = neg_count / len(words)
        neutral = 1.0 - (positive + negative)
        
        return {
            'positive': positive,
            'negative': negative,
            'neutral': neutral
        }
    
    def get_combined_sentiment(self, text):
        """
        Get combined sentiment from all available models.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Combined sentiment scores and metadata
        """
        # Preprocess text
        processed_text = self.preprocessor.preprocess_for_sentiment(text)
        
        # Get sentiments from different models
        vader_scores = self.get_vader_sentiment(processed_text)
        finbert_scores = self.get_finbert_sentiment(processed_text)
        rule_scores = self.get_rule_based_sentiment(processed_text)
        
        # Combine scores with different weights
        # FinBERT has highest weight since it's domain-specific
        weights = {
            'finbert': 0.6,
            'vader': 0.3,
            'rule': 0.1
        }
        
        # Normalize weights if some models are unavailable
        if not self.use_finbert:
            weights = {'finbert': 0.0, 'vader': 0.75, 'rule': 0.25}
        if not self.use_vader:
            weights = {'finbert': 0.8, 'vader': 0.0, 'rule': 0.2}
        if not self.use_finbert and not self.use_vader:
            weights = {'finbert': 0.0, 'vader': 0.0, 'rule': 1.0}
        
        # Calculate combined sentiment
        # Map VADER scores to match other models (VADER uses 'pos', 'neg', 'neu')
        vader_positive = vader_scores.get('pos', vader_scores.get('positive', 0.0))
        vader_negative = vader_scores.get('neg', vader_scores.get('negative', 0.0))
        vader_neutral = vader_scores.get('neu', vader_scores.get('neutral', 0.0))
        
        positive = (
            weights['finbert'] * finbert_scores['positive'] +
            weights['vader'] * vader_positive +
            weights['rule'] * rule_scores['positive']
        )
        
        negative = (
            weights['finbert'] * finbert_scores['negative'] +
            weights['vader'] * vader_negative +
            weights['rule'] * rule_scores['negative']
        )
        
        neutral = (
            weights['finbert'] * finbert_scores['neutral'] +
            weights['vader'] * vader_neutral +
            weights['rule'] * rule_scores['neutral']
        )
        
        # Calculate compound score similar to VADER's approach
        compound = positive - negative
        
        # Normalize to between -1 and 1
        if abs(compound) > 0:
            if compound > 0:
                compound = min(compound, 1.0)
            else:
                compound = max(compound, -1.0)
        
        return {
            'compound': compound,
            'positive': positive,
            'negative': negative,
            'neutral': neutral,
            'vader': vader_scores,
            'finbert': finbert_scores,
            'rule': rule_scores
        }
    
    def get_sentiment_label(self, compound):
        """
        Convert compound score to sentiment label.
        
        Args:
            compound (float): Compound sentiment score
            
        Returns:
            str: Sentiment label ('positive', 'negative', or 'neutral')
        """
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_news_batch(self, news_articles=None, days=1):
        """
        Analyze sentiment for a batch of news articles.
        
        Args:
            news_articles (list): List of news articles to analyze
            days (int): If news_articles is None, analyze articles from the last N days
            
        Returns:
            list: List of sentiment results
        """
        if news_articles is None:
            # Fetch recent news from the database
            start_date = datetime.datetime.now() - datetime.timedelta(days=days)
            news_articles = self.db.get_news_articles(start_date=start_date)
        
        results = []
        
        for article in news_articles:
            # Combine headline and summary for analysis
            text = f"{article['headline']} {article['summary']}"
            
            sentiment = self.get_combined_sentiment(text)
            label = self.get_sentiment_label(sentiment['compound'])
            
            result = {
                'news_id': article['news_id'],
                'symbol': article['symbol'],
                'source': article['source'],
                'timestamp': article.get('published_at', datetime.datetime.now()),
                'content_type': 'news',
                'compound_score': sentiment['compound'],
                'positive_score': sentiment['positive'],
                'negative_score': sentiment['negative'],
                'neutral_score': sentiment['neutral'],
                'sentiment_label': label,
                'analyzed_at': datetime.datetime.now()
            }
            
            results.append(result)
        
        # Store results in the database
        if results:
            self.db.insert_sentiment_scores(results)
        
        return results
    
    def get_sentiment_summary(self, symbol, days=1):
        """
        Get sentiment summary for a specific stock.
        
        Args:
            symbol (str): Stock ticker symbol
            days (int): Number of days to include
            
        Returns:
            dict: Sentiment summary statistics
        """
        # Get sentiment data from the database
        start_date = datetime.datetime.now() - datetime.timedelta(days=days)
        sentiment_data = self.db.get_sentiment_scores(symbol=symbol, start_date=start_date)
        
        if not sentiment_data:
            return {
                'symbol': symbol,
                'data_points': 0,
                'average_sentiment': 0,
                'positive_ratio': 0,
                'negative_ratio': 0,
                'neutral_ratio': 0,
                'news_sentiment': 0
            }
        
        df = pd.DataFrame(sentiment_data)
        
        # Calculate overall sentiment statistics
        avg_sentiment = df['compound_score'].mean()
        sentiment_counts = df['sentiment_label'].value_counts()
        total = len(df)
        
        positive_ratio = sentiment_counts.get('positive', 0) / total if total > 0 else 0
        negative_ratio = sentiment_counts.get('negative', 0) / total if total > 0 else 0
        neutral_ratio = sentiment_counts.get('neutral', 0) / total if total > 0 else 0
        
        # Get news sentiment
        news_sentiment = df[df['content_type'] == 'news']['compound_score'].mean() \
            if not df[df['content_type'] == 'news'].empty else 0
        
        return {
            'symbol': symbol,
            'data_points': total,
            'average_sentiment': avg_sentiment,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'neutral_ratio': neutral_ratio,
            'news_sentiment': news_sentiment
        }

# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    
    sample_texts = [
        "Apple's revenue surpassed expectations, showing strong growth in services.",
        "Tesla missed earnings targets as production challenges continue to impact deliveries.",
        "Microsoft announced new cloud services, maintaining its competitive position."
    ]
    
    for text in sample_texts:
        sentiment = analyzer.get_combined_sentiment(text)
        label = analyzer.get_sentiment_label(sentiment['compound'])
        
        print(f"Text: {text}")
        print(f"Sentiment: {label} (Score: {sentiment['compound']:.4f})")
        print(f"Positive: {sentiment['positive']:.4f}, Negative: {sentiment['negative']:.4f}, Neutral: {sentiment['neutral']:.4f}")
        print("-" * 80)