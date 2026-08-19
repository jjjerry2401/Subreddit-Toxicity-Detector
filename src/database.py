"""
Database Module

This module handles all database operations for storing and retrieving
comments, toxicity scores, and aggregated data.
"""

import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager

class Database:
    """
    A class to handle database operations for the toxicity detection system.
    """
    
    def __init__(self, db_path='reddit.db'):
        """
        Initialize the database connection.
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.create_tables()
        
    @contextmanager
    def get_connection(self):
        """
        Create a context manager for database connections.
        
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Allows accessing columns by name
        try:
            yield conn
        finally:
            conn.close()
            
    def create_tables(self):
        """
        Create necessary tables if they don't exist.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create comments table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comment_text TEXT NOT NULL,
                    subreddit TEXT NOT NULL,
                    toxicity_score REAL NOT NULL,
                    status TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create aggregated metrics table for time windows
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS window_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    window_start DATETIME NOT NULL,
                    window_end DATETIME NOT NULL,
                    total_comments INTEGER DEFAULT 0,
                    toxic_comments INTEGER DEFAULT 0,
                    toxicity_percentage REAL DEFAULT 0,
                    avg_toxicity_score REAL DEFAULT 0,
                    comments_per_minute REAL DEFAULT 0,
                    anomaly_status TEXT DEFAULT 'NORMAL'
                )
            ''')
            
            # Create drama alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drama_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    window_start DATETIME NOT NULL,
                    window_end DATETIME NOT NULL,
                    current_toxicity REAL NOT NULL,
                    historical_mean REAL NOT NULL,
                    historical_std REAL NOT NULL,
                    threshold REAL NOT NULL,
                    alert_reason TEXT,
                    status TEXT DEFAULT 'DRAMA_ALERT'
                )
            ''')

            # Store reviewed predictions so they can be included in later training.
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS labeled_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    comment_text TEXT NOT NULL,
                    label INTEGER NOT NULL,
                    source TEXT DEFAULT 'dashboard',
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            print("Database tables created successfully.")

    def insert_feedback(self, comment_text, label, source='dashboard'):
        """Save a human-reviewed label for a comment."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO labeled_feedback (comment_text, label, source)
                VALUES (?, ?, ?)
            ''', (comment_text, int(label), source))
            conn.commit()
            return cursor.lastrowid

    def get_labeled_feedback(self):
        """Return all human-reviewed comments for model training."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT comment_text, label FROM labeled_feedback
                ORDER BY timestamp ASC
            ''')
            return [dict(row) for row in cursor.fetchall()]

    def get_feedback_count(self):
        """Return the number of human-reviewed comments."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) AS count FROM labeled_feedback')
            return cursor.fetchone()['count']
            
    def insert_comment(self, comment_text, subreddit, toxicity_score, status):
        """
        Insert a new comment into the database.
        
        Args:
            comment_text (str): The comment text
            subreddit (str): The subreddit name
            toxicity_score (float): The toxicity score (0-1)
            status (str): 'TOXIC' or 'SAFE'
            
        Returns:
            int: The ID of the inserted row
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO comments (comment_text, subreddit, toxicity_score, status)
                VALUES (?, ?, ?, ?)
            ''', (comment_text, subreddit, toxicity_score, status))
            conn.commit()
            return cursor.lastrowid
            
    def get_recent_comments(self, limit=100):
        """
        Get the most recent comments.
        
        Args:
            limit (int): Maximum number of comments to return
            
        Returns:
            list: List of comments with their data
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM comments 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
            
    def get_comments_by_time(self, start_time, end_time):
        """
        Get comments within a time range.
        
        Args:
            start_time (str): Start timestamp
            end_time (str): End timestamp
            
        Returns:
            list: List of comments
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM comments 
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            ''', (start_time, end_time))
            return [dict(row) for row in cursor.fetchall()]
            
    def insert_window_metrics(self, window_start, window_end, total_comments, 
                            toxic_comments, toxicity_percentage, avg_toxicity_score,
                            comments_per_minute, anomaly_status='NORMAL'):
        """
        Insert aggregated metrics for a time window.
        
        Args:
            window_start (str): Window start timestamp
            window_end (str): Window end timestamp
            total_comments (int): Total comments in window
            toxic_comments (int): Toxic comments in window
            toxicity_percentage (float): Percentage of toxic comments
            avg_toxicity_score (float): Average toxicity score
            comments_per_minute (float): Comments per minute
            anomaly_status (str): 'NORMAL', 'WARNING', or 'DRAMA_ALERT'
            
        Returns:
            int: The ID of the inserted row
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO window_metrics 
                (window_start, window_end, total_comments, toxic_comments, 
                 toxicity_percentage, avg_toxicity_score, comments_per_minute, 
                 anomaly_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (window_start, window_end, total_comments, toxic_comments,
                  toxicity_percentage, avg_toxicity_score, comments_per_minute,
                  anomaly_status))
            conn.commit()
            return cursor.lastrowid
            
    def get_window_metrics(self, limit=100):
        """
        Get recent window metrics.
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of window metrics
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM window_metrics 
                ORDER BY window_start DESC 
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
            
    def insert_drama_alert(self, window_start, window_end, current_toxicity,
                          historical_mean, historical_std, threshold, alert_reason):
        """
        Insert a drama alert into the database.
        
        Args:
            window_start (str): Window start timestamp
            window_end (str): Window end timestamp
            current_toxicity (float): Current toxicity percentage
            historical_mean (float): Historical mean toxicity
            historical_std (float): Historical standard deviation
            threshold (float): The threshold used
            alert_reason (str): Description of why the alert was triggered
            
        Returns:
            int: The ID of the inserted row
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO drama_alerts 
                (window_start, window_end, current_toxicity, historical_mean, 
                 historical_std, threshold, alert_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (window_start, window_end, current_toxicity, historical_mean,
                  historical_std, threshold, alert_reason))
            conn.commit()
            return cursor.lastrowid
            
    def get_drama_alerts(self, limit=50):
        """
        Get recent drama alerts.
        
        Args:
            limit (int): Maximum number of alerts to return
            
        Returns:
            list: List of drama alerts
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM drama_alerts 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            return [dict(row) for row in cursor.fetchall()]
            
    def get_total_comments(self):
        """
        Get total number of comments in the database.
        
        Returns:
            int: Total comment count
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM comments')
            return cursor.fetchone()['count']
            
    def get_toxic_comments(self):
        """
        Get number of toxic comments in the database.
        
        Returns:
            int: Toxic comment count
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) as count FROM comments WHERE status="TOXIC"')
            return cursor.fetchone()['count']
            
    def get_average_toxicity_score(self):
        """
        Get average toxicity score across all comments.
        
        Returns:
            float: Average toxicity score
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT AVG(toxicity_score) as avg_score FROM comments')
            result = cursor.fetchone()
            return result['avg_score'] if result['avg_score'] is not None else 0.0
            
    def get_latest_window_status(self):
        """
        Get the status of the most recent time window.
        
        Returns:
            dict: Latest window metrics or None if no data
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM window_metrics 
                ORDER BY window_start DESC 
                LIMIT 1
            ''')
            result = cursor.fetchone()
            return dict(result) if result else None
            
    def get_comments_for_export(self, limit=None):
        """
        Get comments for export/display purposes.
        
        Args:
            limit (int, optional): Maximum number of comments to return
            
        Returns:
            list: List of comments with their data
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT comment_text, toxicity_score, status, timestamp 
                FROM comments 
                ORDER BY timestamp DESC
            '''
            if limit:
                query += f' LIMIT {limit}'
            cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]


# Create a singleton instance for easy import
db = Database()


# Testing function
if __name__ == "__main__":
    # Test the database
    print("Testing Database Module...")
    print("="*50)
    
    # Insert a test comment
    test_id = db.insert_comment(
        "This is a moron!",
        "TestSubreddit",
        0.6,
        "TOXIC"
    )
    print(f"Inserted test comment with ID: {test_id}")
    
    # Get total comments
    total = db.get_total_comments()
    print(f"Total comments: {total}")
    
    # Get toxic count
    toxic = db.get_toxic_comments()
    print(f"Toxic comments: {toxic}")
    
    # Get average toxicity
    avg = db.get_average_toxicity_score()
    print(f"Average toxicity score: {avg:.4f}")
    
    print("\nDatabase test complete!")