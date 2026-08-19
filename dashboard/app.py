"""
Streamlit Dashboard

This module creates a web dashboard for monitoring community toxicity in real-time.
"""

import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import streamlit as st

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database import db
from src.anomaly_detector import anomaly_detector
from src.detector import detector
from src.train_model import ToxicityModel

# Page configuration
st.set_page_config(
    page_title="Community Toxicity Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-normal {
        color: #00cc66;
        font-weight: bold;
        font-size: 1.5em;
    }
    .status-warning {
        color: #ffaa00;
        font-weight: bold;
        font-size: 1.5em;
    }
    .status-drama {
        color: #ff3333;
        font-weight: bold;
        font-size: 1.5em;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .drama-alert-box {
        background-color: #ffebee;
        border: 3px solid #ff3333;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        animation: pulse 2s infinite;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Real-Time Community Anomalies")
st.subheader("NLP-Powered Toxicity Detection & Drama Alert System")

# Sidebar
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Auto-refresh
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
    
    # Subreddit display
    st.info(f"📌 Monitoring: r/{os.getenv('SUBREDDIT', 'AskReddit')}")
    
    # Stats
    st.header("📊 Statistics")
    
    # Load data
    @st.cache_data(ttl=5)
    def load_data():
        comments = db.get_recent_comments(limit=1000)
        return pd.DataFrame(comments) if comments else pd.DataFrame()
    
    @st.cache_data(ttl=5)
    def load_window_metrics():
        windows = db.get_window_metrics(limit=50)
        return pd.DataFrame(windows) if windows else pd.DataFrame()
    
    @st.cache_data(ttl=5)
    def load_alerts():
        alerts = db.get_drama_alerts(limit=20)
        return pd.DataFrame(alerts) if alerts else pd.DataFrame()
    
    df = load_data()
    window_df = load_window_metrics()
    alerts_df = load_alerts()

    st.caption(f"Reviewed examples: {db.get_feedback_count()}")
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

    if st.button("🧠 Retrain model"):
        with st.spinner("Training with the original dataset and reviewed comments..."):
            ToxicityModel().train()
            detector.load_model()
        st.cache_data.clear()
        st.success("Model retrained successfully.")

    export_df = pd.DataFrame(db.get_comments_for_export())
    if not export_df.empty:
        st.download_button(
            "⬇️ Export comments CSV",
            export_df.to_csv(index=False).encode('utf-8'),
            file_name="toxicity_comments.csv",
            mime="text/csv"
        )
    
    # Display metrics
    if not df.empty:
        total_comments = len(df)
        toxic_comments = len(df[df['status'] == 'TOXIC'])
        toxicity_percentage = (toxic_comments / total_comments) * 100 if total_comments > 0 else 0
        avg_score = df['toxicity_score'].mean() if not df.empty else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💬 Total Comments", total_comments)
        with col2:
            st.metric("⚠️ Toxic Comments", toxic_comments)
        
        st.metric("📈 Toxicity %", f"{toxicity_percentage:.1f}%")
        st.metric("📊 Avg Score", f"{avg_score:.3f}")
    else:
        st.warning("No data available yet. Waiting for comments...")

# Main content - Status
current_status = anomaly_detector.get_current_status()
status = current_status.get('status', 'NORMAL')
toxicity_pct = current_status.get('toxicity_percentage', 0)

# Status display
status_colors = {
    'NORMAL': ('🟢', 'NORMAL', 'normal'),
    'WARNING': ('🟡', 'WARNING', 'warning'),
    'DRAMA_ALERT': ('🔴', '🚨 DRAMA ALERT!', 'drama')
}

emoji, status_text, status_class = status_colors.get(status, ('⚪', 'UNKNOWN', 'normal'))

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if status == 'DRAMA_ALERT':
        st.markdown(f"""
            <div class="drama-alert-box">
                <h1 style="text-align: center; color: #ff3333;">🚨 DRAMA ALERT! 🚨</h1>
                <p style="text-align: center; font-size: 1.2em;">
                    Toxicity spike detected! Community conflict in progress.
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="text-align: center; padding: 20px;">
                <span class="status-{status_class}" style="font-size: 3em;">{emoji}</span>
                <h2 style="margin-top: 0;">Current Status: <span class="status-{status_class}">{status_text}</span></h2>
                <p style="font-size: 1.2em;">Toxicity Level: {toxicity_pct:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

# Latest alert reason
if status != 'NORMAL' and not window_df.empty:
    latest = window_df.iloc[0] if not window_df.empty else None
    if latest and latest.get('anomaly_status') != 'NORMAL':
        st.info(f"ℹ️ {latest.get('reason', 'No reason provided')}")

# Charts
st.header("📈 Toxicity Trends")

if not window_df.empty:
    # Reverse to show chronological order
    window_df = window_df.sort_values('window_start', ascending=True)
    
    # Create three columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Toxicity percentage over time
        fig1 = px.line(
            window_df, 
            x='window_start', 
            y='toxicity_percentage',
            title='Toxicity Percentage Over Time',
            labels={'toxicity_percentage': 'Toxicity %', 'window_start': 'Time'}
        )
        
        # Add threshold line if available
        # Add status color coding
        colors = {'NORMAL': 'green', 'WARNING': 'orange', 'DRAMA_ALERT': 'red'}
        fig1.add_scatter(
            x=window_df['window_start'],
            y=window_df['toxicity_percentage'],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Toxicity %'
        )
        
        # Color points by status
        status_colors = window_df['anomaly_status'].map(colors)
        fig1.add_scatter(
            x=window_df['window_start'],
            y=window_df['toxicity_percentage'],
            mode='markers',
            marker=dict(color=status_colors, size=8),
            name='Status'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Comments per minute
        fig2 = px.bar(
            window_df,
            x='window_start',
            y='comments_per_minute',
            title='Comments Per Minute',
            labels={'comments_per_minute': 'Comments/Min', 'window_start': 'Time'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Third chart - Average toxicity score
    fig3 = px.line(
        window_df,
        x='window_start',
        y='avg_toxicity_score',
        title='Average Toxicity Score Over Time',
        labels={'avg_toxicity_score': 'Avg Toxicity Score', 'window_start': 'Time'}
    )
    st.plotly_chart(fig3, use_container_width=True)
    
else:
    st.info("📊 Not enough data for charts yet. Waiting for more comments...")

# Interactive prediction tools
st.header("🧪 Analyze Comments")
st.caption("Review predictions to create labeled examples for the next training run.")
analysis_tab, batch_tab = st.tabs(["Single comment", "Batch comments"])

with analysis_tab:
    with st.form("single_comment_form"):
        comment_text = st.text_area("Comment", placeholder="Paste a comment to analyze...", height=100)
        subreddit = st.text_input("Subreddit", value=os.getenv('SUBREDDIT', 'AskReddit'))
        submitted = st.form_submit_button("🔍 Analyze comment")

    if submitted:
        if not comment_text.strip():
            st.warning("Enter a comment before analyzing.")
        else:
            st.session_state['last_prediction'] = detector.process_comment(comment_text, subreddit)

    prediction = st.session_state.get('last_prediction')
    if prediction:
        if prediction.get('error'):
            st.error(prediction['error'])
        else:
            st.metric("Prediction", f"{prediction['status']} ({prediction['score']:.1%})")
            feedback_col1, feedback_col2 = st.columns(2)
            with feedback_col1:
                if st.button("✅ Mark safe", key="mark_safe"):
                    db.insert_feedback(comment_text, 0)
                    st.success("Saved as a safe training example.")
            with feedback_col2:
                if st.button("⚠️ Mark toxic", key="mark_toxic"):
                    db.insert_feedback(comment_text, 1)
                    st.success("Saved as a toxic training example.")

with batch_tab:
    with st.form("batch_comment_form"):
        batch_text = st.text_area(
            "One comment per line",
            placeholder="Helpful advice\nThis is awful",
            height=140
        )
        batch_subreddit = st.text_input("Batch subreddit", value=os.getenv('SUBREDDIT', 'AskReddit'))
        batch_submitted = st.form_submit_button("▶️ Analyze batch")

    if batch_submitted:
        comments = [line.strip() for line in batch_text.splitlines() if line.strip()]
        if not comments:
            st.warning("Add at least one comment.")
        else:
            results = [detector.process_comment(comment, batch_subreddit) for comment in comments]
            batch_results = pd.DataFrame([
                {'Comment': comment, 'Status': result.get('status'), 'Toxicity Score': result.get('score', 0)}
                for comment, result in zip(comments, results)
            ])
            st.dataframe(batch_results, use_container_width=True)

# Recent comments table
st.header("💬 Recent Comments")

if not df.empty:
    # Display recent comments with status coloring
    display_df = df.head(20).copy()
    
    # Function to color status
    def color_status(status):
        if status == 'TOXIC':
            return '🔴 TOXIC'
        else:
            return '🟢 SAFE'
    
    display_df['display_status'] = display_df['status'].apply(color_status)
    
    # Select columns to display
    table_df = display_df[['comment_text', 'toxicity_score', 'display_status', 'timestamp']]
    table_df.columns = ['Comment', 'Toxicity Score', 'Status', 'Timestamp']
    
    st.dataframe(table_df, use_container_width=True)
else:
    st.info("💬 No comments available yet.")

# Drama Alerts History
st.header("🚨 Drama Alerts History")

if not alerts_df.empty:
    # Format alerts for display
    alerts_display = alerts_df[['timestamp', 'window_start', 'window_end', 'current_toxicity', 'alert_reason']].head(10)
    alerts_display.columns = ['Timestamp', 'Window Start', 'Window End', 'Toxicity %', 'Reason']
    alerts_display['Toxicity %'] = alerts_display['Toxicity %'].round(1)
    
    st.dataframe(alerts_display, use_container_width=True)
else:
    st.info("No drama alerts recorded yet.")

# Auto-refresh logic
if auto_refresh:
    st.empty()
    st.cache_data.clear()
    import time
    time.sleep(5)
    st.rerun()

# Footer
st.markdown("---")
st.caption("🔬 Built with Python, Scikit-learn, Streamlit, and PRAW | Real-Time Community Anomalies")