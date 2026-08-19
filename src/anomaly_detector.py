"""
Anomaly Detection Module

This module detects anomalies in community toxicity levels over time.
It uses rolling statistics to identify sudden spikes in toxicity.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.database import db

class AnomalyDetector:
    """
    A class to detect anomalies in community toxicity levels.
    """
    
    def __init__(self, window_minutes=5, anomaly_threshold=2.5, min_history=3):
        """
        Initialize the anomaly detector.
        
        Args:
            window_minutes (int): Time window in minutes for aggregation
            anomaly_threshold (float): Number of standard deviations for anomaly detection
            min_history (int): Minimum number of windows needed for detection
        """
        self.window_minutes = window_minutes
        self.anomaly_threshold = anomaly_threshold
        self.min_history = min_history
        self.window_data = deque(maxlen=100)  # Store recent window metrics
        
    def aggregate_window(self, comments):
        """
        Aggregate comments for a time window.
        
        Args:
            comments (list): List of comment dictionaries with 'toxicity_score' and 'status'
            
        Returns:
            dict: Aggregated metrics for the window
        """
        if not comments:
            return None
            
        total = len(comments)
        toxic = sum(1 for c in comments if c['status'] == 'TOXIC')
        toxicity_percentage = (toxic / total) * 100 if total > 0 else 0
        avg_score = sum(c['toxicity_score'] for c in comments) / total if total > 0 else 0
        
        return {
            'total_comments': total,
            'toxic_comments': toxic,
            'toxicity_percentage': toxicity_percentage,
            'avg_toxicity_score': avg_score,
            'comments_per_minute': total / self.window_minutes
        }
    
    def get_rolling_stats(self, history):
        """
        Calculate rolling mean and standard deviation of toxicity percentages.
        
        Args:
            history (list): List of window metrics with 'toxicity_percentage'
            
        Returns:
            tuple: (mean, std) of historical toxicity percentages
        """
        if len(history) < self.min_history:
            return None, None
            
        toxicity_values = [w['toxicity_percentage'] for w in history]
        mean = np.mean(toxicity_values)
        std = np.std(toxicity_values)
        return mean, std
    
    def detect_anomaly(self, current_window, historical_windows):
        """
        Detect if the current window represents an anomaly.
        
        Args:
            current_window (dict): Current window metrics
            historical_windows (list): Historical window metrics
            
        Returns:
            tuple: (is_anomaly, status, reason, stats)
        """
        if current_window is None:
            return False, 'NORMAL', 'No data available', None
            
        # Get historical statistics
        mean, std = self.get_rolling_stats(historical_windows)
        
        if mean is None or std is None:
            return False, 'NORMAL', 'Insufficient historical data', {
                'mean': None,
                'std': None,
                'threshold': None
            }
        
        # Calculate threshold
        threshold = mean + self.anomaly_threshold * std
        current_toxicity = current_window['toxicity_percentage']
        
        # Determine status
        is_anomaly = current_toxicity > threshold
        status = 'NORMAL'
        reason = 'Toxicity is within normal range'
        
        if is_anomaly:
            status = 'DRAMA_ALERT'
            reason = (
                f"Drama Alert: Toxicity increased from {mean:.1f}% "
                f"to {current_toxicity:.1f}% in the last {self.window_minutes} minutes "
                f"(Threshold: {threshold:.1f}%)"
            )
        elif current_toxicity > mean + (self.anomaly_threshold / 2) * std:
            status = 'WARNING'
            reason = (
                f"Warning: Toxicity is rising at {current_toxicity:.1f}% "
                f"(historical mean: {mean:.1f}%, std: {std:.1f}%)"
            )
        
        stats = {
            'mean': mean,
            'std': std,
            'threshold': threshold,
            'current_toxicity': current_toxicity
        }
        
        return is_anomaly, status, reason, stats
    
    def process_comments(self, comments, window_start, window_end):
        """
        Process a batch of comments for a time window.
        
        Args:
            comments (list): List of comment dictionaries
            window_start (datetime): Window start time
            window_end (datetime): Window end time
            
        Returns:
            dict: Processing results including metrics and anomaly status
        """
        if not comments:
            return None
            
        # Aggregate window metrics
        metrics = self.aggregate_window(comments)
        if metrics is None:
            return None
            
        # Add window timestamps
        metrics['window_start'] = window_start
        metrics['window_end'] = window_end
        
        # Get historical windows from database
        historical_windows = db.get_window_metrics(limit=20)
        # Reverse to get chronological order
        historical_windows = list(reversed(historical_windows))
        
        # Detect anomaly
        is_anomaly, status, reason, stats = self.detect_anomaly(
            metrics, historical_windows
        )
        
        # Store results
        result = {
            'metrics': metrics,
            'status': status,
            'reason': reason,
            'is_anomaly': is_anomaly,
            'stats': stats
        }
        
        # Store in database
        try:
            window_id = db.insert_window_metrics(
                window_start.isoformat(),
                window_end.isoformat(),
                metrics['total_comments'],
                metrics['toxic_comments'],
                metrics['toxicity_percentage'],
                metrics['avg_toxicity_score'],
                metrics['comments_per_minute'],
                status
            )
            
            # If drama alert, store it separately
            if is_anomaly and stats:
                db.insert_drama_alert(
                    window_start.isoformat(),
                    window_end.isoformat(),
                    metrics['toxicity_percentage'],
                    stats['mean'],
                    stats['std'],
                    stats['threshold'],
                    reason
                )
                
        except Exception as e:
            print(f"Error storing window metrics: {e}")
            
        return result
    
    def get_current_status(self):
        """
        Get the current community status.
        
        Returns:
            dict: Current status information
        """
        latest = db.get_latest_window_status()
        if latest:
            return {
                'status': latest['anomaly_status'],
                'toxicity_percentage': latest['toxicity_percentage'],
                'window_start': latest['window_start'],
                'window_end': latest['window_end']
            }
        return {'status': 'NORMAL', 'toxicity_percentage': 0}


# Create a singleton instance
anomaly_detector = AnomalyDetector()


# Testing function
if __name__ == "__main__":
    # Test the anomaly detector with sample data
    print("Testing Anomaly Detector...")
    print("="*50)
    
    # Create some sample window metrics
    sample_windows = []
    
    # Simulate 10 normal windows (15-20% toxicity)
    for i in range(10):
        sample_windows.append({
            'toxicity_percentage': np.random.normal(18, 5)
        })
    
    # Add some anomalous windows
    sample_windows.append({'toxicity_percentage': 75})
    sample_windows.append({'toxicity_percentage': 82})
    
    # Test detection
    for i, window in enumerate(sample_windows):
        historical = sample_windows[:i]  # Use previous windows as history
        is_anomaly, status, reason, stats = anomaly_detector.detect_anomaly(
            window, historical
        )
        print(f"Window {i+1}: Toxicity={window['toxicity_percentage']:.1f}%")
        print(f"  Status: {status}")
        if stats and stats['mean'] is not None:
            print(f"  Historical: mean={stats['mean']:.1f}%, std={stats['std']:.1f}%")
            print(f"  Threshold: {stats['threshold']:.1f}%")
        print(f"  Reason: {reason}")
        print("-" * 50)
    
    print("Anomaly detector test complete!")