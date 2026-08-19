"""
Text Preprocessing Module

This module handles cleaning and preprocessing of text data
for the toxicity detection pipeline.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class TextPreprocessor:
    """
    A class to handle text preprocessing for toxicity detection.
    """
    
    def __init__(self):
        """Initialize the preprocessor with stopwords."""
        self.stop_words = set(stopwords.words('english'))
        # Add custom stopwords that might not be useful for toxicity detection
        self.stop_words.update(['http', 'https', 'www', 'com', 'org', 'net'])
        
    def clean_text(self, text):
        """
        Clean and preprocess text for toxicity detection.
        
        Args:
            text (str): The raw text to clean
            
        Returns:
            str: The cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags (keep the text but remove @ and #)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and punctuation (keep letters, numbers, and spaces)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers (optional - might be useful for context)
        # text = re.sub(r'\d+', '', text)
        
        return text
    
    def tokenize_text(self, text):
        """
        Tokenize text into words.
        
        Args:
            text (str): The cleaned text
            
        Returns:
            list: List of tokens
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens
    
    def preprocess(self, text):
        """
        Complete preprocessing pipeline.
        
        Args:
            text (str): The raw text
            
        Returns:
            str: Preprocessed text ready for vectorization
        """
        # Clean the text
        cleaned = self.clean_text(text)
        
        # Tokenize
        tokens = self.tokenize_text(cleaned)
        
        # Join tokens back into a string for vectorization
        return ' '.join(tokens)
    
    def preprocess_batch(self, texts):
        """
        Preprocess a batch of texts.
        
        Args:
            texts (list): List of raw texts
            
        Returns:
            list: List of preprocessed texts
        """
        return [self.preprocess(text) for text in texts]


# Create a singleton instance for easy import
preprocessor = TextPreprocessor()


# Testing function
if __name__ == "__main__":
    # Test the preprocessor
    test_texts = [
        "You are absolutely useless! http://fakeurl.com",
        "I love this community!!! #blessed",
        "This is the worst thing I've ever seen @user123",
        "",
        None
    ]
    
    print("Testing Text Preprocessor:")
    print("-" * 50)
    
    for text in test_texts:
        cleaned = preprocessor.clean_text(text)
        preprocessed = preprocessor.preprocess(text)
        print(f"Original: {text}")
        print(f"Cleaned:  {cleaned}")
        print(f"Final:    {preprocessed}")
        print("-" * 50)