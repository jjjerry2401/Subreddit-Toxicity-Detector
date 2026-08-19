"""
Toxicity Detector Module

This module loads the trained model and uses it to predict toxicity
in comments in real-time.
"""

import os
import sys
import joblib
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocessor
from src.database import db

class ToxicityDetector:
    """
    A class to detect toxicity in text using a trained ML model.
    """
    
    def __init__(self, model_path='models/toxicity_model.pkl', 
                 vectorizer_path='models/vectorizer.pkl'):
        """
        Initialize the detector and load the model.
        
        Args:
            model_path (str): Path to the trained model file
            vectorizer_path (str): Path to the TF-IDF vectorizer file
        """
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.classes = [0, 1]
        self.is_loaded = False
        
        # Load model
        self.load_model()
        
    def load_model(self):
        """
        Load the trained model and vectorizer.
        
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                self.model = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                classes_path = os.path.join(os.path.dirname(self.model_path), 'model_classes.pkl')
                if os.path.exists(classes_path):
                    self.classes = joblib.load(classes_path)
                else:
                    self.classes = list(getattr(self.model, 'classes_', [0, 1]))
                self.is_loaded = True
                print("Model and vectorizer loaded successfully.")
                return True
            else:
                print(f"Model files not found. Please train the model first.")
                print(f"Looking for: {self.model_path} and {self.vectorizer_path}")
                return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, text):
        """
        Predict whether a text is toxic.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Prediction results including score and status
        """
        if not self.is_loaded:
            return {
                'error': 'Model not loaded',
                'status': 'ERROR',
                'score': 0.0
            }
        
        if not text or not isinstance(text, str):
            return {
                'status': 'SAFE',
                'score': 0.0,
                'error': 'Empty or invalid text'
            }
        
        try:
            # Preprocess text
            cleaned_text = preprocessor.preprocess(text)
            
            # If text is empty after preprocessing, treat as safe
            if not cleaned_text:
                return {
                    'status': 'SAFE',
                    'score': 0.0,
                    'processed_text': cleaned_text
                }
            
            # Vectorize
            vectorized = self.vectorizer.transform([cleaned_text])
            
            # Get prediction probability
            proba = self.model.predict_proba(vectorized)[0]
            
            # Get toxicity score (probability of being toxic)
            toxic_index = self.classes.index(1) if 1 in self.classes else 0
            toxicity_score = float(proba[toxic_index])
            
            # Determine status
            status = 'TOXIC' if toxicity_score >= 0.5 else 'SAFE'
            
            return {
                'status': status,
                'score': toxicity_score,
                'processed_text': cleaned_text
            }
            
        except Exception as e:
            print(f"Error predicting toxicity: {e}")
            return {
                'error': str(e),
                'status': 'ERROR',
                'score': 0.0
            }
    
    def process_comment(self, comment_text, subreddit):
        """
        Process a comment through the full pipeline.
        
        Args:
            comment_text (str): The comment text
            subreddit (str): The subreddit name
            
        Returns:
            dict: Full processing results
        """
        # Get prediction
        prediction = self.predict(comment_text)
        
        if prediction.get('error'):
            return prediction
        
        # Store in database
        try:
            comment_id = db.insert_comment(
                comment_text[:1000],  # Truncate long comments
                subreddit,
                prediction['score'],
                prediction['status']
            )
            prediction['comment_id'] = comment_id
        except Exception as e:
            print(f"Error storing comment: {e}")
            prediction['db_error'] = str(e)
        
        # Add timestamp
        prediction['timestamp'] = datetime.now().isoformat()
        
        return prediction


# Create a singleton instance
detector = ToxicityDetector()


# Testing function
if __name__ == "__main__":
    print("Testing Toxicity Detector...")
    print("="*50)
    
    # Test comments
    test_comments = [
        "You are really helpful and kind!",
        "This is the worst thing ever, you're useless!",
        "I appreciate your time and effort.",
        "I hope you fail miserably!",
        "Let's work together on this project.",
        "You have no idea what you're doing!"
    ]
    
    for comment in test_comments:
        result = detector.predict(comment)
        print(f"Comment: {comment}")
        print(f"  Status: {result['status']}")
        print(f"  Score: {result['score']:.4f}")
        if 'error' in result:
            print(f"  Error: {result['error']}")
        print("-" * 30)