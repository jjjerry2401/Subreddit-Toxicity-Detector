"""
Model Training Module

This module trains a machine learning model to detect toxicity in text.
It uses TF-IDF vectorization and Logistic Regression.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import preprocessor
from src.database import db

class ToxicityModel:
    """
    A class to handle training and saving the toxicity detection model.
    """
    
    def __init__(self, data_path='data/dataset.csv'):
        """
        Initialize the model trainer.
        
        Args:
            data_path (str): Path to the dataset CSV file
        """
        self.data_path = data_path
        self.model = None
        self.vectorizer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Create directories if they don't exist
        os.makedirs('models', exist_ok=True)
        
    def load_data(self):
        """
        Load and preprocess the dataset.
        
        Returns:
            tuple: (X, y) features and labels
        """
        if not os.path.exists(self.data_path):
            print(f"Error: Dataset not found at {self.data_path}")
            print("Please ensure data/dataset.csv exists.")
            return None, None
        
        # Load the original labeled dataset.
        df = pd.read_csv(self.data_path)
        print(f"Loaded {len(df)} samples from {self.data_path}")
        
        # Check for missing values
        if df.isnull().any().any():
            print("Warning: Missing values found. Dropping rows with missing values.")
            df = df.dropna()
        
        # Append reviewed dashboard examples so future training learns from use.
        feedback = pd.DataFrame(db.get_labeled_feedback())
        if not feedback.empty:
            feedback.columns = ['comment', 'label']
            df = pd.concat([df[['comment', 'label']], feedback], ignore_index=True)
            df = df.drop_duplicates(subset=['comment', 'label'])
            print(f"Including {len(feedback)} reviewed comments from the dashboard")

        # Preprocess the complete dataset, including reviewed examples.
        print("Preprocessing text...")
        df['processed_comment'] = df['comment'].apply(preprocessor.preprocess)

        # Remove any empty processed texts
        df = df[df['processed_comment'].str.len() > 0]
        print(f"After preprocessing: {len(df)} samples")

        # Get features and labels
        X = df['processed_comment'].values
        y = df['label'].values
        
        return X, y
    
    def train(self):
        """
        Train the toxicity detection model.
        """
        # Load data
        X, y = self.load_data()
        if X is None or y is None:
            return

        if len(set(y)) < 2:
            print("Training requires both safe (0) and toxic (1) examples.")
            return
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        # Create TF-IDF vectorizer
        print("Creating TF-IDF vectorizer...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True
        )
        
        # Transform training data
        print("Vectorizing training data...")
        X_train_vectorized = self.vectorizer.fit_transform(self.X_train)
        
        # Train model
        print("Training Logistic Regression model...")
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        self.model.fit(X_train_vectorized, self.y_train)

        # Persist the class order used by predict_proba for robust inference.
        joblib.dump(self.model.classes_.tolist(), 'models/model_classes.pkl')
        
        # Evaluate model
        self.evaluate()
        
        # Save model and vectorizer
        self.save_model()
        
    def evaluate(self):
        """
        Evaluate the trained model on test data.
        """
        if self.model is None or self.vectorizer is None:
            print("Model not trained yet.")
            return
        
        # Transform test data
        X_test_vectorized = self.vectorizer.transform(self.X_test)
        
        # Make predictions
        y_pred = self.model.predict(X_test_vectorized)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Non-Toxic', 'Toxic']))
        print("="*50)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_model(self):
        """
        Save the trained model and vectorizer.
        """
        if self.model is None or self.vectorizer is None:
            print("No model to save.")
            return
        
        # Save model
        model_path = 'models/toxicity_model.pkl'
        vectorizer_path = 'models/vectorizer.pkl'
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
        
    def load_model(self):
        """
        Load a pre-trained model and vectorizer.
        
        Returns:
            tuple: (model, vectorizer)
        """
        model_path = 'models/toxicity_model.pkl'
        vectorizer_path = 'models/vectorizer.pkl'
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print(f"Model files not found. Please train the model first.")
            return None, None
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        self.model = model
        self.vectorizer = vectorizer
        
        print(f"Model loaded from: {model_path}")
        print(f"Vectorizer loaded from: {vectorizer_path}")
        
        return model, vectorizer


def main():
    """
    Main function to train the model.
    """
    print("Starting Toxicity Model Training...")
    print("="*50)
    
    # Create trainer instance
    trainer = ToxicityModel()
    
    # Train the model
    trainer.train()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()