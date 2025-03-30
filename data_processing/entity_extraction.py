"""
Entity extraction module for financial text data.
Extracts named entities such as companies, people, locations, and financial terms.
"""

import os
import sys
import pandas as pd
import re
import logging
import datetime
import nltk

# Try to import spaCy for NER, with fallback to NLTK
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    # Ensure NLTK resources are available
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('maxent_ne_chunker/english_ace_multiclass.pickle')
        nltk.data.find('words/english.pickle')
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('punkt')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        nltk.download('averaged_perceptron_tagger')

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

class EntityExtractor:
    """
    Class for extracting named entities from financial texts.
    """
    
    def __init__(self, use_spacy=True):
        """
        Initialize the entity extractor.
        
        Args:
            use_spacy (bool): Whether to use spaCy for NER (if available)
        """
        self.preprocessor = TextPreprocessor()
        self.db = DatabaseConnector()
        
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        
        # Initialize spaCy model if available
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy NER model initialized")
            except Exception as e:
                logger.error(f"Failed to initialize spaCy model: {e}")
                self.use_spacy = False
        
        # Financial entity patterns (for regex-based extraction)
        self.ticker_pattern = re.compile(r'\$([A-Z]{1,5})\b')
        self.percentage_pattern = re.compile(r'(\d+(\.\d+)?\s*%)|(percent|percentage)')
        self.money_pattern = re.compile(r'\$\d+(\.\d+)?(\s*(million|billion|trillion))?|\d+(\.\d+)?\s*(dollars|USD)')
        
        # Common financial entities
        self.financial_entities = {
            'market_indices': {
                'S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000', 'NYSE',
                'FTSE', 'Nikkei', 'DAX', 'CAC 40', 'Hang Seng'
            },
            'regulators': {
                'SEC', 'Federal Reserve', 'Fed', 'FINRA', 'CFTC', 'OCC',
                'FDIC', 'Treasury', 'IRS', 'ECB', 'Bank of England'
            },
            'financial_terms': {
                'IPO', 'M&A', 'merger', 'acquisition', 'dividend', 'earnings',
                'revenue', 'profit', 'loss', 'guidance', 'forecast', 'outlook',
                'downgrade', 'upgrade', 'buy', 'sell', 'hold', 'overweight',
                'underweight', 'market cap', 'valuation', 'P/E ratio'
            }
        }
    
    def extract_entities_spacy(self, text):
        """
        Extract entities using spaCy NER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary of extracted entities by type
        """
        if not self.use_spacy:
            return {}
        
        doc = self.nlp(text)
        
        entities = {
            'ORG': [],  # Organizations
            'PERSON': [],  # People
            'GPE': [],  # Geopolitical entities (countries, cities)
            'LOC': [],  # Locations
            'DATE': [],  # Dates
            'MONEY': [],  # Monetary values
            'PERCENT': []  # Percentages
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                # Avoid duplicates
                if ent.text not in entities[ent.label_]:
                    entities[ent.label_].append(ent.text)
        
        return entities
    
    def extract_entities_nltk(self, text):
        """
        Extract entities using NLTK NER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary of extracted entities by type
        """
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        named_entities = nltk.ne_chunk(pos_tags)
        
        entities = {
            'ORGANIZATION': [],
            'PERSON': [],
            'LOCATION': [],
            'DATE': [],
            'MONEY': [],
            'PERCENT': []
        }
        
        # Extract named entities
        for chunk in named_entities:
            if hasattr(chunk, 'label'):
                entity_type = chunk.label()
                entity_text = ' '.join(c[0] for c in chunk)
                
                if entity_type in entities and entity_text not in entities[entity_type]:
                    entities[entity_type].append(entity_text)
        
        return entities
    
    def extract_financial_entities(self, text):
        """
        Extract financial-specific entities using regex and pattern matching.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary of extracted financial entities
        """
        financial_entities = {
            'tickers': [],
            'percentages': [],
            'monetary_values': [],
            'market_indices': [],
            'regulators': [],
            'financial_terms': []
        }
        
        # Extract stock tickers (e.g., $AAPL)
        tickers = self.ticker_pattern.findall(text)
        financial_entities['tickers'] = list(set(tickers))
        
        # Extract percentages
        percentages = self.percentage_pattern.findall(text)
        financial_entities['percentages'] = [p[0] for p in percentages if p[0]]
        
        # Extract monetary values
        monetary = self.money_pattern.findall(text)
        financial_entities['monetary_values'] = [m[0] for m in monetary if m[0]]
        
        # Extract known financial entities
        lower_text = text.lower()
        for category, terms in self.financial_entities.items():
            found = []
            for term in terms:
                if term.lower() in lower_text:
                    found.append(term)
            financial_entities[category] = found
        
        return financial_entities
    
    def get_combined_entities(self, text):
        """
        Get combined entities from all extraction methods.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Combined extracted entities
        """
        # Preprocess text
        processed_text = self.preprocessor.normalize_text(text)
        
        # Extract entities using different methods
        spacy_entities = self.extract_entities_spacy(processed_text) if self.use_spacy else {}
        nltk_entities = self.extract_entities_nltk(processed_text) if not self.use_spacy else {}
        financial_entities = self.extract_financial_entities(processed_text)
        
        # Combine entities
        combined = {
            'organizations': spacy_entities.get('ORG', []) or nltk_entities.get('ORGANIZATION', []),
            'persons': spacy_entities.get('PERSON', []) or nltk_entities.get('PERSON', []),
            'locations': (spacy_entities.get('GPE', []) + spacy_entities.get('LOC', [])) or nltk_entities.get('LOCATION', []),
            'dates': spacy_entities.get('DATE', []) or nltk_entities.get('DATE', []),
            'monetary_values': spacy_entities.get('MONEY', []) or nltk_entities.get('MONEY', []) or financial_entities['monetary_values'],
            'percentages': spacy_entities.get('PERCENT', []) or nltk_entities.get('PERCENT', []) or financial_entities['percentages'],
            'tickers': financial_entities['tickers'],
            'market_indices': financial_entities['market_indices'],
            'regulators': financial_entities['regulators'],
            'financial_terms': financial_entities['financial_terms']
        }
        
        # Remove duplicates and empty lists
        for key, value in combined.items():
            combined[key] = list(set(value))
        
        return combined
    
    def process_news_batch(self, news_articles=None, days=1):
        """
        Process a batch of news articles for entity extraction.
        
        Args:
            news_articles (list): List of news articles to process
            days (int): If news_articles is None, process articles from the last N days
            
        Returns:
            list: List of entity extraction results
        """
        if news_articles is None:
            # Fetch recent news from the database
            start_date = datetime.datetime.now() - datetime.timedelta(days=days)
            news_articles = self.db.get_news_articles(start_date=start_date)
        
        results = []
        db_entities = []
        
        for article in news_articles:
            # Combine headline and summary for analysis
            text = f"{article['headline']} {article['summary']}"
            
            entities = self.get_combined_entities(text)
            
            result = {
                'reference_id': article['news_id'],
                'symbol': article['symbol'],
                'source': article['source'],
                'timestamp': article.get('published_at', datetime.datetime.now()),
                'content_type': 'news',
                'entities': entities,
                'extracted_at': datetime.datetime.now()
            }
            
            results.append(result)
            
            # Format entities for database insertion
            for entity_type, entity_list in entities.items():
                for entity_value in entity_list:
                    db_entities.append({
                        'reference_id': article['news_id'],
                        'source': article['source'],
                        'entity_type': entity_type,
                        'entity_value': entity_value,
                        'entity_text': entity_value,
                        'timestamp': article.get('published_at', datetime.datetime.now()),
                        'extracted_at': datetime.datetime.now()
                    })
        
        # Store results in the database
        if db_entities:
            self.db.insert_entities(db_entities)
        
        return results
    
    def process_social_batch(self, social_posts=None, days=1):
        """
        Process a batch of social media posts for entity extraction.
        
        Args:
            social_posts (list): List of social media posts to process
            days (int): If social_posts is None, process posts from the last N days
            
        Returns:
            list: List of entity extraction results
        """
        if social_posts is None:
            # Fetch recent posts from the database
            start_date = datetime.datetime.now() - datetime.timedelta(days=days)
            social_posts = self.db.get_social_media_posts(start_date=start_date)
        
        results = []
        db_entities = []
        
        for post in social_posts:
            # Combine title and content for analysis
            text = f"{post['title']} {post['content']}"
            
            entities = self.get_combined_entities(text)
            
            result = {
                'reference_id': post['post_id'],
                'symbol': post['symbol'],
                'source': f"{post['platform']}-{post['subreddit']}" if post['subreddit'] else post['platform'],
                'timestamp': post.get('created_at', datetime.datetime.now()),
                'content_type': 'social',
                'entities': entities,
                'extracted_at': datetime.datetime.now()
            }
            
            results.append(result)
            
            # Format entities for database insertion
            for entity_type, entity_list in entities.items():
                for entity_value in entity_list:
                    db_entities.append({
                        'reference_id': post['post_id'],
                        'source': result['source'],
                        'entity_type': entity_type,
                        'entity_value': entity_value,
                        'entity_text': entity_value,
                        'timestamp': post.get('created_at', datetime.datetime.now()),
                        'extracted_at': datetime.datetime.now()
                    })
        
        # Store results in the database
        if db_entities:
            self.db.insert_entities(db_entities)
        
        return results
    
    def get_entity_summary(self, symbol, days=1):
        """
        Get entity summary for a specific stock.
        
        Args:
            symbol (str): Stock ticker symbol
            days (int): Number of days to include
            
        Returns:
            dict: Entity summary statistics
        """
        # Get entity data from the database
        start_date = datetime.datetime.now() - datetime.timedelta(days=days)
        entity_data = self.db.get_entities(symbol=symbol, start_date=start_date)
        
        if not entity_data:
            return {
                'symbol': symbol,
                'data_points': 0,
                'top_organizations': [],
                'top_persons': [],
                'top_locations': [],
                'top_financial_terms': [],
                'related_tickers': []
            }
        
        # Aggregate entities
        organizations = []
        persons = []
        locations = []
        financial_terms = []
        tickers = []
        
        for data in entity_data:
            entities = data['entities']
            organizations.extend(entities.get('organizations', []))
            persons.extend(entities.get('persons', []))
            locations.extend(entities.get('locations', []))
            financial_terms.extend(entities.get('financial_terms', []))
            tickers.extend(entities.get('tickers', []))
        
        # Count frequencies
        org_counts = pd.Series(organizations).value_counts().head(5).to_dict()
        person_counts = pd.Series(persons).value_counts().head(5).to_dict()
        location_counts = pd.Series(locations).value_counts().head(5).to_dict()
        term_counts = pd.Series(financial_terms).value_counts().head(5).to_dict()
        ticker_counts = pd.Series(tickers).value_counts().head(5).to_dict()
        
        return {
            'symbol': symbol,
            'data_points': len(entity_data),
            'top_organizations': [{'name': k, 'count': v} for k, v in org_counts.items()],
            'top_persons': [{'name': k, 'count': v} for k, v in person_counts.items()],
            'top_locations': [{'name': k, 'count': v} for k, v in location_counts.items()],
            'top_financial_terms': [{'term': k, 'count': v} for k, v in term_counts.items()],
            'related_tickers': [{'ticker': k, 'count': v} for k, v in ticker_counts.items()]
        }