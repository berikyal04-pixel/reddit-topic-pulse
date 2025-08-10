import streamlit as st
import pandas as pd
import time
from reddit_coffee_pulse import load_config, init_reddit, pull_posts, sentiment_scores, cluster_texts, generate_report

st.set_page_config(page_title="Reddit Topic Pulse", layout="wide")

st.title("📊 Reddit Topic Pulse")
st.write("Enter keywords, select subreddits, and get sentiment + theme analysis from Reddit posts.")

# User input
subreddits_input = st.text_input("Subreddits (comma-separated)", "coffee, barista, espresso, sustainability, zerowaste")
queries_input = st.text_input("Search keywords (comma-separated)", "coffee, sustainable pods, zero waste coffee")
since_days = st.slider("Look back (days)", 1, 30, 7)
k_clusters = st.slider("Number of themes (clusters)", 2, 10, 6)
max_posts = st.slider("Max posts to fetch", 50, 1000, 300)

if st.button("Run Analysis"):
    with st.spinner("Fetching data from Reddit..."):
        subreddits = [s.strip() for s in subreddits_input.split(",") if s.strip()]
        queries = [q.strip() for q in queries_input.split(",") if q.strip()]
        
        reddit = init_reddit()
        df = pull_posts(reddit, subreddits, queries, since_days, max_posts)
        
        if df.empty:
            st.warning("No posts found for your criteria.")
        else:
            df["sentiment"] = sentiment_scores(df["text"].tolist())
            labels, cluster_names = cluster_texts(df["text"].tolist(), k_clusters)
            df["cluster"] = labels
            
            # Show results
            st.subheader("Theme Breakdown")
            theme_counts = df["cluster"].value_counts().sort_index()
            for idx, name in enumerate(cluster_names):
                st.write(f"**Theme {idx+1}: {name}** — {theme_counts.get(idx, 0)} posts")
            
            st.subheader("Sentiment")
            pos = (df["sentiment"] == 1).sum()
            neu = (df["sentiment"] == 0).sum()
            neg = (df["sentiment"] == -1).sum()
            st.write(f"Positive: {pos}")
            st.write(f"Neutral: {neu}")
            st.write(f"Negative: {neg}")
            
            st.subheader("Posts Data")
            st.dataframe(df[["date", "subreddit", "title", "query", "sentiment", "cluster"]])
