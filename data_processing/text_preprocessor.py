"""
Text preprocessing module for cleaning and preparing text data for NLP.
"""

import re
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt_tab/english/')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    
# Log successful download of NLTK resources
logger.info("NLTK resources successfully verified/downloaded")

class TextPreprocessor:
    """Class for preprocessing and cleaning text data."""
    
    def __init__(self, language='english'):
        """
        Initialize the text preprocessor.
        
        Args:
            language (str): Language for stopwords (default: 'english')
        """
        self.stop_words = set(stopwords.words(language))
        
        # Add financial-specific stopwords
        self.financial_stopwords = {
            'stock', 'stocks', 'market', 'markets', 'trade', 'trading',
            'invest', 'investing', 'investment', 'investor', 'investors',
            'share', 'shares', 'price', 'prices', 'value', 'growth',
            'profit', 'profits', 'loss', 'losses', 'quarter', 'quarterly'
        }
        
        # Add common ticker-related terms to avoid confusion
        self.ticker_related = {
            'nasdaq', 'nyse', 'dow', 'djia', 'sp500', 's&p', 'etf', 'ipo'
        }
    
    def normalize_text(self, text):
        """
        Normalize text: lowercase, remove accents, extra spaces, etc.
        
        Args:
            text (str): Text to normalize
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove accents
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_punctuation(self, text):
        """
        Remove punctuation from text.
        
        Args:
            text (str): Text to process
            
        Returns:
            str: Text without punctuation
        """
        # Keep $ for stock tickers like $AAPL
        translator = str.maketrans('', '', string.punctuation.replace('$', ''))
        return text.translate(translator)
    
    def remove_stopwords(self, text, keep_financial_terms=True):
        """
        Remove stopwords from text.
        
        Args:
            text (str): Text to process
            keep_financial_terms (bool): Whether to keep financial terms
            
        Returns:
            str: Text without stopwords
        """
        tokens = word_tokenize(text)
        
        if keep_financial_terms:
            filtered_tokens = [
                token for token in tokens 
                if token not in self.stop_words or token in self.financial_stopwords
            ]
        else:
            # Remove all stopwords including financial terms
            all_stopwords = self.stop_words.union(self.financial_stopwords)
            filtered_tokens = [token for token in tokens if token not in all_stopwords]
        
        return ' '.join(filtered_tokens)
    
    def preprocess_for_sentiment(self, text):
        """
        Preprocess text for sentiment analysis.
        
        Args:
            text (str): Raw text to process
            
        Returns:
            str: Processed text ready for sentiment analysis
        """
        if not text:
            return ""
        
        # Normalize
        text = self.normalize_text(text)
        
        # Handle stock tickers (preserve $AAPL format)
        tickers = re.findall(r'\$[A-Za-z]+', text)
        
        # Replace tickers with placeholders
        for i, ticker in enumerate(tickers):
            text = text.replace(ticker, f' TICKER{i} ')
        
        # Remove punctuation
        text = self.remove_punctuation(text)
        
        # Restore tickers
        for i, ticker in enumerate(tickers):
            text = text.replace(f'TICKER{i}', ticker)
        
        # Remove stopwords but keep financial terms
        text = self.remove_stopwords(text, keep_financial_terms=True)
        
        return text
    
    def preprocess_for_entity_extraction(self, text):
        """
        Preprocess text for entity extraction.
        
        Args:
            text (str): Raw text to process
            
        Returns:
            str: Processed text ready for entity extraction
        """
        if not text:
            return ""
        
        # Normalize but keep case for named entity recognition
        text = text.replace('\n', ' ')
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def process_batch(self, texts, process_type='sentiment'):
        """
        Process a batch of texts.
        
        Args:
            texts (list): List of texts to process
            process_type (str): Type of processing ('sentiment' or 'entity')
            
        Returns:
            list: List of processed texts
        """
        processed_texts = []
        
        for text in texts:
            if process_type == 'sentiment':
                processed_text = self.preprocess_for_sentiment(text)
            elif process_type == 'entity':
                processed_text = self.preprocess_for_entity_extraction(text)
            else:
                processed_text = self.normalize_text(text)
            
            processed_texts.append(processed_text)
        
        return processed_texts

# Example usage
if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    
    sample_text = """
    $AAPL is up 2% today after their earnings call. The company reported $89.5B
    in revenue, beating estimates. CEO Tim Cook mentioned strong iPhone 13 sales
    and services growth. https://apple.com/investor
    """
    
    print("Original text:")
    print(sample_text)
    
    print("\nPreprocessed for sentiment analysis:")
    print(preprocessor.preprocess_for_sentiment(sample_text))
    
    print("\nPreprocessed for entity extraction:")
    print(preprocessor.preprocess_for_entity_extraction(sample_text))