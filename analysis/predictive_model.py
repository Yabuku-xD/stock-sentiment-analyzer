"""
Predictive model for stock price movements based on sentiment.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
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

class PredictiveModel:
    """
    Class for building and evaluating predictive models based on sentiment data.
    """
    
    def __init__(self):
        """Initialize the predictive model."""
        self.db = DatabaseConnector()
        self.models = {}
        self.features = [
            'compound_score', 'positive_score', 'negative_score', 'neutral_score',
            'lagged_sentiment_1', 'lagged_sentiment_2', 'lagged_sentiment_3', 
            'prev_price_change_1', 'prev_price_change_2', 'prev_price_change_3',
            'sentiment_ma_5', 'sentiment_ma_10', 'sentiment_std_5',
            'sentiment_slope', 'news_count', 'social_count'
        ]
    
    def _prepare_features(self, symbol, start_date, end_date, window_size=60):
        """
        Prepare features for the predictive model.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for data collection
            end_date (datetime): End date for data collection
            window_size (int): Window size in minutes
            
        Returns:
            pandas.DataFrame: DataFrame with features and target variable
        """
        # Get extra data for lag calculations
        extended_start_date = start_date - datetime.timedelta(days=3)
        
        # Get stock price data
        price_data = self.db.get_stock_prices(
            symbol=symbol, 
            start_date=extended_start_date, 
            end_date=end_date
        )
        
        # Get sentiment data
        sentiment_data = self.db.get_sentiment_scores(
            symbol=symbol, 
            start_date=extended_start_date, 
            end_date=end_date
        )
        
        # Get news count data
        news_count = self.db.execute_query(
            f"""
            SELECT 
                DATE_TRUNC('hour', published_at) as hour,
                COUNT(*) as news_count
            FROM {self.db.tables['news_articles']}
            WHERE 
                symbol = %(symbol)s AND
                published_at BETWEEN %(start_date)s AND %(end_date)s
            GROUP BY DATE_TRUNC('hour', published_at)
            """,
            {'symbol': symbol, 'start_date': extended_start_date, 'end_date': end_date}
        )
        
        # Get social media post count data
        social_count = self.db.execute_query(
            f"""
            SELECT 
                DATE_TRUNC('hour', created_at) as hour,
                COUNT(*) as social_count
            FROM {self.db.tables['social_media_posts']}
            WHERE 
                symbol = %(symbol)s AND
                created_at BETWEEN %(start_date)s AND %(end_date)s
            GROUP BY DATE_TRUNC('hour', created_at)
            """,
            {'symbol': symbol, 'start_date': extended_start_date, 'end_date': end_date}
        )
        
        if not price_data or not sentiment_data:
            logger.warning(f"Insufficient data for {symbol} in the specified time range")
            return None
        
        # Convert to DataFrames
        prices_df = pd.DataFrame(price_data)
        sentiment_df = pd.DataFrame(sentiment_data)
        news_df = pd.DataFrame(news_count, columns=['hour', 'news_count'])
        social_df = pd.DataFrame(social_count, columns=['hour', 'social_count'])
        
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
        sentiment_windows.rename(columns={'reference_id': 'sentiment_count'}, inplace=True)
        
        # Calculate price changes
        price_windows['price_change'] = (price_windows['close'] - price_windows['open']) / price_windows['open'] * 100
        
        # Add news and social counts
        news_df['window'] = pd.to_datetime(news_df['hour']).dt.floor(f'{window_size}min')
        social_df['window'] = pd.to_datetime(social_df['hour']).dt.floor(f'{window_size}min')
        
        news_agg = news_df.groupby('window').agg({'news_count': 'sum'}).reset_index()
        social_agg = social_df.groupby('window').agg({'social_count': 'sum'}).reset_index()
        
        # Merge all data
        df = price_windows.merge(sentiment_windows, on='window', how='left')
        df = df.merge(news_agg, on='window', how='left')
        df = df.merge(social_agg, on='window', how='left')
        
        # Fill missing values
        df['news_count'] = df['news_count'].fillna(0)
        df['social_count'] = df['social_count'].fillna(0)
        df['compound_score'] = df['compound_score'].fillna(0)
        df['positive_score'] = df['positive_score'].fillna(0)
        df['negative_score'] = df['negative_score'].fillna(0)
        df['neutral_score'] = df['neutral_score'].fillna(0)
        
        # Create target variable: Direction of next period's price change
        df['next_price_change'] = df['price_change'].shift(-1)
        df['price_up'] = (df['next_price_change'] > 0).astype(int)
        
        # Create lagged features
        for lag in range(1, 4):
            df[f'lagged_sentiment_{lag}'] = df['compound_score'].shift(lag)
            df[f'prev_price_change_{lag}'] = df['price_change'].shift(lag)
        
        # Create moving averages and other technical features
        df['sentiment_ma_5'] = df['compound_score'].rolling(window=5).mean()
        df['sentiment_ma_10'] = df['compound_score'].rolling(window=10).mean()
        df['sentiment_std_5'] = df['compound_score'].rolling(window=5).std()
        
        # Calculate sentiment slope (rate of change)
        df['sentiment_slope'] = (df['compound_score'] - df['lagged_sentiment_3']) / 3
        
        # Drop rows with NaN (due to lag calculations)
        df = df.dropna()
        
        # Keep only data from the original start date
        df = df[df['window'] >= pd.Timestamp(start_date)]
        
        return df
    
    def build_model(self, symbol, start_date, end_date, window_size=60, model_type='random_forest'):
        """
        Build and train a predictive model.
        
        Args:
            symbol (str): Stock ticker symbol
            start_date (datetime): Start date for training data
            end_date (datetime): End date for training data
            window_size (int): Window size in minutes
            model_type (str): Type of model to build
            
        Returns:
            dict: Dictionary with model and evaluation metrics
        """
        # Prepare features
        df = self._prepare_features(symbol, start_date, end_date, window_size)
        
        if df is None or len(df) < 20:
            logger.warning(f"Insufficient data for {symbol} to build a model")
            return {
                'symbol': symbol,
                'model_type': model_type,
                'success': False,
                'message': "Insufficient data to build model"
            }
        
        # Select features and target
        X = df[self.features]
        y = df['price_up']
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build model
        model = None
        param_grid = {}
        
        if model_type == 'random_forest':
            model = RandomForestClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif model_type == 'logistic_regression':
            model = LogisticRegression(random_state=42, max_iter=1000)
            param_grid = {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        elif model_type == 'svm':
            model = SVC(random_state=42, probability=True)
            param_grid = {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 1.0]
            }
        elif model_type == 'xgboost':
            model = xgb.XGBClassifier(random_state=42)
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
        else:
            logger.error(f"Unknown model type: {model_type}")
            return {
                'symbol': symbol,
                'model_type': model_type,
                'success': False,
                'message': f"Unknown model type: {model_type}"
            }
        
        # Perform grid search for hyperparameter tuning
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Make predictions
        y_pred = best_model.predict(X_test_scaled)
        
        # Evaluate model
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Feature importance
        feature_importances = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = dict(zip(self.features, best_model.feature_importances_))
        elif model_type == 'logistic_regression':
            feature_importances = dict(zip(self.features, best_model.coef_[0]))
        elif model_type == 'svm' and best_model.kernel == 'linear':
            feature_importances = dict(zip(self.features, best_model.coef_[0]))
        
        # Save model
        model_id = f"{symbol}_{model_type}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{model_id}.joblib")
        scaler_path = os.path.join(model_dir, f"{model_id}_scaler.joblib")
        
        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Store in instance
        self.models[symbol] = {
            'model': best_model,
            'scaler': scaler,
            'features': self.features,
            'model_id': model_id,
            'model_path': model_path,
            'scaler_path': scaler_path
        }
        
        # Return results
        results = {
            'symbol': symbol,
            'model_type': model_type,
            'success': True,
            'model_id': model_id,
            'best_params': grid_search.best_params_,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist(),
            'feature_importances': feature_importances,
            'training_data_points': len(X_train),
            'test_data_points': len(X_test),
            'positive_class_ratio': y.mean()
        }
        
        return results
    
    def predict(self, symbol, features_dict):
        """
        Make a prediction using the trained model.
        
        Args:
            symbol (str): Stock ticker symbol
            features_dict (dict): Dictionary of feature values
            
        Returns:
            dict: Prediction results
        """
        if symbol not in self.models:
            logger.error(f"No model found for {symbol}")
            return {
                'symbol': symbol,
                'success': False,
                'message': f"No model found for {symbol}"
            }
        
        # Get model
        model_info = self.models[symbol]
        model = model_info['model']
        scaler = model_info['scaler']
        features = model_info['features']
        
        # Ensure all features are present
        for feature in features:
            if feature not in features_dict:
                features_dict[feature] = 0
        
        # Prepare features
        X = pd.DataFrame([features_dict], columns=features)
        X_scaled = scaler.transform(X)
        
        # Make prediction
        probability = model.predict_proba(X_scaled)[0, 1]
        prediction = int(probability >= 0.5)
        
        return {
            'symbol': symbol,
            'success': True,
            'prediction': prediction,
            'probability': probability,
            'timestamp': datetime.datetime.now()
        }
    
    def load_model(self, model_path, scaler_path):
        """
        Load a saved model.
        
        Args:
            model_path (str): Path to saved model
            scaler_path (str): Path to saved scaler
            
        Returns:
            tuple: Loaded model and scaler
        """
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    
    def build_models_for_all_stocks(self, days=30, window_size=60, model_type='random_forest'):
        """
        Build models for all target stocks.
        
        Args:
            days (int): Number of days of data to use
            window_size (int): Window size in minutes
            model_type (str): Type of model to build
            
        Returns:
            dict: Dictionary with model results for all stocks
        """
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        all_results = {}
        
        for stock in TARGET_STOCKS:
            symbol = stock['symbol']
            logger.info(f"Building model for {symbol}")
            
            results = self.build_model(
                symbol,
                start_date,
                end_date,
                window_size,
                model_type
            )
            
            all_results[symbol] = results
        
        return all_results
    
    def plot_feature_importance(self, model_results):
        """
        Plot feature importances from a model.
        
        Args:
            model_results (dict): Model results dictionary
            
        Returns:
            matplotlib.figure.Figure: Matplotlib figure with feature importance plot
        """
        if not model_results['success'] or 'feature_importances' not in model_results:
            logger.warning("No feature importances available")
            return None
        
        feature_importances = model_results['feature_importances']
        
        # Sort features by importance
        sorted_features = sorted(
            feature_importances.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        feature_names = [f[0] for f in sorted_features]
        importance_values = [f[1] for f in sorted_features]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['green' if i > 0 else 'red' for i in importance_values]
        ax.barh(feature_names, importance_values, color=colors)
        
        ax.set_xlabel('Feature Importance')
        ax.set_title(f'Feature Importance for {model_results["symbol"]} ({model_results["model_type"]})')
        ax.grid(True, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, model_results):
        """
        Plot confusion matrix from model results.
        
        Args:
            model_results (dict): Model results dictionary
            
        Returns:
            matplotlib.figure.Figure: Matplotlib figure with confusion matrix plot
        """
        if not model_results['success'] or 'confusion_matrix' not in model_results:
            logger.warning("No confusion matrix available")
            return None
        
        conf_matrix = np.array(model_results['confusion_matrix'])
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 8))
        
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Predicted Down', 'Predicted Up'],
            yticklabels=['Actual Down', 'Actual Up']
        )
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'Confusion Matrix for {model_results["symbol"]} ({model_results["model_type"]})')
        
        plt.tight_layout()
        return fig
    
    def generate_model_report(self, model_results):
        """
        Generate a report for a model.
        
        Args:
            model_results (dict): Model results dictionary
            
        Returns:
            str: Markdown-formatted report
        """
        if not model_results['success']:
            return f"# Model for {model_results['symbol']} Failed\n\n{model_results['message']}"
        
        # Format confusion matrix
        conf_matrix = np.array(model_results['confusion_matrix'])
        tn, fp, fn, tp = conf_matrix.ravel()
        
        # Sort feature importances
        feature_importances = model_results['feature_importances']
        sorted_features = sorted(
            feature_importances.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        # Create report
        report = [
            f"# Predictive Model Report: {model_results['symbol']}",
            f"**Model Type:** {model_results['model_type']}",
            f"**Model ID:** {model_results['model_id']}",
            f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Model Performance",
            f"- **Accuracy:** {model_results['accuracy']:.4f}",
            f"- **Precision:** {model_results['precision']:.4f}",
            f"- **Recall:** {model_results['recall']:.4f}",
            f"- **F1 Score:** {model_results['f1_score']:.4f}",
            "\n## Confusion Matrix",
            "```",
            "                 Predicted Down  Predicted Up",
            f"Actual Down     {tn}            {fp}",
            f"Actual Up       {fn}            {tp}",
            "```",
            f"\n- **True Positives (TP):** {tp} - Correctly predicted price increase",
            f"- **False Positives (FP):** {fp} - Incorrectly predicted price increase",
            f"- **True Negatives (TN):** {tn} - Correctly predicted price decrease",
            f"- **False Negatives (FN):** {fn} - Incorrectly predicted price decrease",
            "\n## Feature Importance",
            "The following features had the most impact on predictions:",
            "\n| Feature | Importance |",
            "|---------|------------|"
        ]
        
        # Add top 10 features
        for feature, importance in sorted_features[:10]:
            report.append(f"| {feature} | {importance:.4f} |")
        
        report.extend([
            "\n## Model Parameters",
            "The model was trained with the following best parameters:"
        ])
        
        for param, value in model_results['best_params'].items():
            report.append(f"- **{param}:** {value}")
        
        report.extend([
            "\n## Training Information",
            f"- **Training Data Points:** {model_results['training_data_points']}",
            f"- **Test Data Points:** {model_results['test_data_points']}",
            f"- **Positive Class Ratio:** {model_results['positive_class_ratio']:.4f} (proportion of price increases)",
            "\n## Usage",
            "To use this model for prediction:",
            "```python",
            "from analysis.predictive_model import PredictiveModel",
            "",
            "model = PredictiveModel()",
            f"model.load_model('{model_results['model_id']}.joblib', '{model_results['model_id']}_scaler.joblib')",
            "",
            "# Prepare features",
            "features = {",
            "    'compound_score': 0.5,",
            "    'positive_score': 0.7,",
            "    'negative_score': 0.1,",
            "    # ... include all required features",
            "}",
            "",
            f"prediction = model.predict('{model_results['symbol']}', features)",
            "print(f\"Prediction: {'Up' if prediction['prediction'] == 1 else 'Down'} with probability {prediction['probability']:.4f}\")",
            "```"
        ])
        
        return "\n".join(report)

if __name__ == "__main__":
    model = PredictiveModel()
    
    # Example usage
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=30)
    
    # Build model for a specific stock
    results = model.build_model('AAPL', start_date, end_date, model_type='random_forest')
    
    if results['success']:
        print(f"Model built successfully for AAPL")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        
        # Generate and save plots
        fig1 = model.plot_feature_importance(results)
        if fig1:
            fig1.savefig('aapl_feature_importance.png')
            print("Feature importance plot saved as aapl_feature_importance.png")
        
        fig2 = model.plot_confusion_matrix(results)
        if fig2:
            fig2.savefig('aapl_confusion_matrix.png')
            print("Confusion matrix plot saved as aapl_confusion_matrix.png")
        
        # Generate report
        report = model.generate_model_report(results)
        with open('aapl_model_report.md', 'w') as f:
            f.write(report)
        print("Model report saved as aapl_model_report.md")
    else:
        print(f"Failed to build model: {results['message']}")