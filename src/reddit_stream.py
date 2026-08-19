"""
Reddit Stream Module

This module connects to the Reddit API and streams comments in real-time.
It also includes a DEMO MODE for testing without API credentials.
"""

import os
import sys
import time
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detector import detector
from src.database import db
from src.anomaly_detector import anomaly_detector

# Load environment variables
load_dotenv()

class RedditStream:
    """
    A class to stream comments from Reddit.
    """
    
    def __init__(self):
        """Initialize the Reddit stream."""
        self.demo_mode = os.getenv('DEMO_MODE', 'true').lower() == 'true'
        self.subreddit = os.getenv('SUBREDDIT', 'AskReddit')
        self.window_minutes = int(os.getenv('WINDOW_MINUTES', '5'))
        self.anomaly_threshold = float(os.getenv('ANOMALY_THRESHOLD', '2.5'))
        
        self.reddit = None
        self.comment_buffer = []
        self.window_start = datetime.now()
        self.comment_count = 0
        
        # Sample comments for demo mode
        self.demo_comments = [
            "This is a great idea!",
            "I love this community!",
            "You are absolutely useless!",
            "Thank you so much!",
            "This is terrible!",
            "I appreciate your help!",
            "What a wonderful day!",
            "You are so stupid!",
            "This is the best subreddit!",
            "Go away, nobody wants you here!",
            "I love this project!",
            "This is complete garbage!",
            "Thanks for your support!",
            "You have no idea what you're doing!",
            "Great job everyone!",
            "This is pathetic!",
            "I'm so happy right now!",
            "You are a disgrace!",
            "Let's work together!",
            "Everyone here is a moron!"
        ]
        
        self.demo_toxicity_levels = [0.1, 0.2, 0.3, 0.1, 0.15, 0.1, 0.12, 0.1]
        self.demo_spike_indices = [7, 8, 9, 10]  # Indices where we'll simulate a spike
        
        if self.demo_mode:
            print("=" * 60)
            print("⚡ DEMO MODE ENABLED ⚡")
            print("Using simulated comments instead of Reddit API")
            print("=" * 60)
        else:
            self.connect_reddit()
    
    def connect_reddit(self):
        """
        Connect to the Reddit API using PRAW.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        try:
            import praw
            
            client_id = os.getenv('CLIENT_ID')
            client_secret = os.getenv('CLIENT_SECRET')
            user_agent = os.getenv('USER_AGENT')
            
            if not all([client_id, client_secret, user_agent]):
                print("❌ Missing Reddit API credentials in .env file")
                print("Please add CLIENT_ID, CLIENT_SECRET, and USER_AGENT")
                print("Falling back to DEMO MODE...")
                self.demo_mode = True
                return False
            
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            
            # Test connection
            self.reddit.user.me()  # This will raise an exception if invalid
            print(f"✅ Successfully connected to Reddit API")
            print(f"📡 Streaming from r/{self.subreddit}")
            return True
            
        except ImportError:
            print("❌ PRAW package not installed. Installing...")
            os.system("pip install praw")
            return self.connect_reddit()
        except Exception as e:
            print(f"❌ Failed to connect to Reddit API: {e}")
            print("Falling back to DEMO MODE...")
            self.demo_mode = True
            return False
    
    def get_comment(self):
        """
        Get a single comment (either from Reddit or demo mode).
        
        Returns:
            dict: Comment data with 'body' and 'subreddit' keys
        """
        if self.demo_mode:
            # Generate a simulated comment
            # Increase toxicity gradually, with a spike
            global demo_counter
            if not hasattr(self, 'demo_counter'):
                self.demo_counter = 0
            
            # Select a comment
            comment_text = random.choice(self.demo_comments)
            subreddit = self.subreddit
            
            # Simulate increasing toxicity over time
            if self.demo_counter < len(self.demo_toxicity_levels):
                level = self.demo_toxicity_levels[self.demo_counter]
            else:
                level = 0.3  # Baseline after demo pattern
            
            # Randomize slightly
            if random.random() < 0.2:
                level = max(0.1, level + random.uniform(-0.1, 0.1))
            
            self.demo_counter += 1
            
            return {
                'body': comment_text,
                'subreddit': subreddit,
                'simulated_toxicity': level
            }
        else:
            # Get real comment from Reddit
            try:
                for comment in self.reddit.subreddit(self.subreddit).stream.comments():
                    return {
                        'body': comment.body,
                        'subreddit': self.subreddit,
                        'id': comment.id,
                        'author': str(comment.author)
                    }
            except Exception as e:
                print(f"Error streaming comments: {e}")
                time.sleep(5)
                return None
    
    def process_comment(self, comment_data):
        """
        Process a single comment.
        
        Args:
            comment_data (dict): Comment data from get_comment()
            
        Returns:
            dict: Processing result
        """
        if comment_data is None:
            return None
        
        # Extract comment text
        comment_text = comment_data.get('body', '')
        subreddit = comment_data.get('subreddit', self.subreddit)
        
        if not comment_text:
            return None
        
        # Process through detector
        result = detector.process_comment(comment_text, subreddit)
        
        if result.get('error'):
            return None
        
        # Add to buffer for window aggregation
        self.comment_buffer.append({
            'toxicity_score': result['score'],
            'status': result['status'],
            'timestamp': result['timestamp']
        })
        self.comment_count += 1
        
        # Print progress
        status_emoji = '🔴' if result['status'] == 'TOXIC' else '🟢'
        print(f"{status_emoji} Comment #{self.comment_count}: {result['status']} (Score: {result['score']:.3f})")
        
        return result
    
    def process_window(self):
        """
        Process the current time window of comments.
        
        Returns:
            dict: Window processing results
        """
        if not self.comment_buffer:
            return None
        
        window_end = datetime.now()
        window_start = self.window_start
        
        # Process the window
        result = anomaly_detector.process_comments(
            self.comment_buffer,
            window_start,
            window_end
        )
        
        if result:
            status_emoji = '🔴' if result['status'] == 'DRAMA_ALERT' else '🟡' if result['status'] == 'WARNING' else '🟢'
            print(f"\n📊 Window Analysis ({self.window_minutes} minutes):")
            print(f"   Comments: {result['metrics']['total_comments']}")
            print(f"   Toxic: {result['metrics']['toxic_comments']} ({result['metrics']['toxicity_percentage']:.1f}%)")
            print(f"   Status: {status_emoji} {result['status']}")
            if result['status'] != 'NORMAL':
                print(f"   Reason: {result['reason']}")
            print("-" * 50)
        
        # Reset for next window
        self.comment_buffer = []
        self.window_start = datetime.now()
        
        return result
    
    def run(self, process_interval=5):
        """
        Run the stream processing loop.
        
        Args:
            process_interval (int): Seconds between comment processing
        """
        print(f"\n🚀 Starting Reddit Stream")
        print(f"   Mode: {'DEMO' if self.demo_mode else 'LIVE'}")
        print(f"   Subreddit: r/{self.subreddit}")
        print(f"   Window: {self.window_minutes} minutes")
        print(f"   Anomaly Threshold: {self.anomaly_threshold}σ")
        print("=" * 60)
        
        print("\nPress Ctrl+C to stop the stream\n")
        
        try:
            while True:
                # Get a comment
                comment_data = self.get_comment()
                
                if comment_data:
                    self.process_comment(comment_data)
                
                # Check if we need to process a window
                time_elapsed = (datetime.now() - self.window_start).total_seconds()
                if time_elapsed >= self.window_minutes * 60:
                    self.process_window()
                
                # Wait a bit before processing next comment
                time.sleep(process_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Stream stopped by user")
            print(f"Total comments processed: {self.comment_count}")
            print("Exiting...")
        except Exception as e:
            print(f"❌ Error in stream loop: {e}")
            time.sleep(5)


def main():
    """
    Main function to run the Reddit stream.
    """
    stream = RedditStream()
    stream.run()


if __name__ == "__main__":
    main()