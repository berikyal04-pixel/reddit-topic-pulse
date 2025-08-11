import os
import re
import time
import argparse
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import praw
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import yaml

# Download sentiment lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

def clean_text(txt: str) -> str:
    """Remove URLs, markdown, non-ASCII, extra spaces."""
    if not isinstance(txt, str):
        return ""
    txt = re.sub(r"http\S+|www\.\S+", " ", txt)
    txt = re.sub(r"\[.*?\]\(.*?\)", " ", txt)
    txt = re.sub(r"[\r\n\t]+", " ", txt)
    txt = re.sub(r"[^\x00-\x7F]+", " ", txt)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt

def percent(n, d):
    return 0.0 if d == 0 else round(100.0 * n / d, 1)

def load_config(path: str):
    """Load YAML config."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def init_reddit():
    """Init Reddit API from Streamlit secrets (cloud) or .env (local)."""
    # Try Streamlit secrets first (in cloud)
    client_id = client_secret = user_agent = None
    try:
        import streamlit as st  # will exist on cloud
        s = getattr(st, "secrets", None)
        if s:
            # Accept either flat keys or nested under "reddit"
            if "reddit" in s:
                client_id = s["reddit"].get("CLIENT_ID")
                client_secret = s["reddit"].get("CLIENT_SECRET")
                user_agent = s["reddit"].get("USER_AGENT")
            else:
                client_id = s.get("CLIENT_ID")
                client_secret = s.get("CLIENT_SECRET")
                user_agent = s.get("USER_AGENT")
    except Exception:
        pass

    # Fallback to .env (local dev)
    if not client_id or not client_secret:
        load_dotenv()
        client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        user_agent = user_agent or os.getenv("REDDIT_USER_AGENT")

    if not client_id or not client_secret:
        raise RuntimeError("Missing Reddit API keys. Set Streamlit secrets or .env variables.")

    return praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent or "CoffeePulse/0.1 by u/berikyal04",
    check_for_async=False
)



def pull_posts(reddit, subreddits, queries, since_days, max_posts):
    """Search Reddit with throttling/backoff and return a DataFrame of posts."""
    import time, re, random

    # Defaults (safe). Can be overridden via Streamlit secrets or env.
    REQUEST_SLEEP = 1.0   # seconds between items
    SEARCH_PAUSE  = 3.0   # seconds between searches
    BACKOFF_BASE  = 15.0  # seconds on generic errors

    # Try to read tuning knobs from Streamlit Cloud secrets first
    try:
        import streamlit as st
        s = getattr(st, "secrets", None)
        if s and "throttle" in s:
            REQUEST_SLEEP = float(s["throttle"].get("REQUEST_SLEEP", REQUEST_SLEEP))
            SEARCH_PAUSE  = float(s["throttle"].get("SEARCH_PAUSE",  SEARCH_PAUSE))
            BACKOFF_BASE  = float(s["throttle"].get("BACKOFF_BASE",  BACKOFF_BASE))
    except Exception:
        pass

    # Or from environment variables (local dev)
    REQUEST_SLEEP = float(os.getenv("REQUEST_SLEEP", REQUEST_SLEEP))
    SEARCH_PAUSE  = float(os.getenv("SEARCH_PAUSE",  SEARCH_PAUSE))
    BACKOFF_BASE  = float(os.getenv("BACKOFF_BASE",  BACKOFF_BASE))

    def parse_wait_seconds(msg: str) -> int:
        """Extract 'try again in X minutes/seconds' if present."""
        m = re.search(r"in\s+(\d+)\s*minutes?", msg, re.I)
        if m: return int(m.group(1)) * 60 + random.randint(5, 15)
        m = re.search(r"in\s+(\d+)\s*seconds?", msg, re.I)
        if m: return int(m.group(1)) + random.randint(1, 5)
        return 0

    rows = []
    since_ts = time.time() - since_days * 86400

    for sub in subreddits:
        sr = reddit.subreddit(sub)
        for q in queries:
            remaining = max_posts - len(rows)
            if remaining <= 0:
                break
            limit = min(100, remaining)  # cap per (sub,query) batch

            try:
                for post in sr.search(q, time_filter="week", sort="new", limit=limit):
                    if post.created_utc < since_ts:
                        continue
                    rows.append({
                        "id": post.id,
                        "subreddit": str(post.subreddit),
                        "title": post.title or "",
                        "selftext": post.selftext or "",
                        "created_utc": post.created_utc,
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "query": q,
                        "url": f"https://www.reddit.com{post.permalink}",
                    })
                    if len(rows) >= max_posts:
                        break
                    time.sleep(REQUEST_SLEEP + random.random() * 0.5)  # gentle per-item sleep

            except Exception as e:
                wait = parse_wait_seconds(str(e)) or BACKOFF_BASE
                print(f"[RATELIMIT] Sleeping {wait:.0f}s after error: {e}")
                time.sleep(wait)

            time.sleep(SEARCH_PAUSE)  # pause between searches

        if len(rows) >= max_posts:
            break

    df = pd.DataFrame(rows).drop_duplicates(subset=["id"])
    if df.empty:
        return df
    df["text"] = (df["title"].fillna("") + " " + df["selftext"].fillna("")).apply(clean_text)
    df = df[df["text"].str.len() > 20]
    df["date"] = pd.to_datetime(df["created_utc"], unit="s")
    return df.reset_index(drop=True)

def sentiment_scores(texts):
    sia = SentimentIntensityAnalyzer()
    scores = []
    for t in texts:
        s = sia.polarity_scores(t)["compound"]
        if s >= 0.2:
            scores.append(1)   # positive
        elif s <= -0.2:
            scores.append(-1)  # negative
        else:
            scores.append(0)   # neutral
    return np.array(scores)

def cluster_texts(texts, k):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    import numpy as np

    # No data?
    if not texts or len(texts) == 0:
        return np.array([], dtype=int), []

    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vec.fit_transform(texts)
    n = X.shape[0]

    # Handle tiny datasets (0 or 1 post)
    if n == 1:
        return np.array([0], dtype=int), ["single-topic"]

    # Keep k within [1, n]
    k = max(1, min(k, n))

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    terms = np.array(vec.get_feature_names_out())
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    cluster_names = []
    for i in range(k):
        top_terms = terms[order_centroids[i][:6]]
        cluster_names.append(", ".join(top_terms))

    return labels, cluster_names


def generate_report(df, cluster_names, outdir):
    # Try to import matplotlib only if available (cloud-safe)
    try:
        import matplotlib.pyplot as plt
        HAVE_PLOTS = True
    except Exception:
        HAVE_PLOTS = False

    os.makedirs(outdir, exist_ok=True)
    n = len(df)
    cluster_counts = df["cluster"].value_counts().sort_index()
    pos = (df["sentiment"] == 1).sum()
    neu = (df["sentiment"] == 0).sum()
    neg = (df["sentiment"] == -1).sum()
    start = df["date"].min().strftime("%b %d")
    end = df["date"].max().strftime("%b %d")

    report_path = os.path.join(outdir, "coffee_pulse_report.md")
    lines = []
    lines.append(f"# Reddit Coffee Pulse — {start}–{end}\n")
    lines.append(f"**Coverage:** {n} public posts across {df['subreddit'].nunique()} subreddits.")
    lines.append("\n## Theme breakdown\n")
    for idx, name in enumerate(cluster_names):
        c = int(cluster_counts.get(idx, 0))
        lines.append(f"- **Theme {idx+1}: {name}** — {c} posts ({percent(c, n)}%)")
    lines.append("\n## Sentiment\n")
    lines.append(f"- Positive: {pos} ({percent(pos, n)}%)")
    lines.append(f"- Neutral: {neu} ({percent(neu, n)}%)")
    lines.append(f"- Negative: {neg} ({percent(neg, n)}%)")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Only make the chart if matplotlib is available
    if HAVE_PLOTS:
        plt.figure()
        ax = cluster_counts.plot(kind="bar")
        ax.set_title("Theme sizes")
        ax.set_xlabel("Theme #")
        ax.set_ylabel("Post count")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "cluster_sizes.png"))
        plt.close()

    df.to_csv(os.path.join(outdir, "posts_with_clusters.csv"), index=False)
    return report_path



def main():
    parser = argparse.ArgumentParser(description="Reddit Coffee Pulse")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    reddit = init_reddit()
    df = pull_posts(reddit, cfg["subreddits"], cfg["queries"], cfg["since_days"], cfg["max_posts"])
    if df.empty:
        print("No posts found.")
        return
    df["sentiment"] = sentiment_scores(df["text"].tolist())
    labels, cluster_names = cluster_texts(df["text"].tolist(), cfg["k_clusters"])
    df["cluster"] = labels
    path = generate_report(df, cluster_names, cfg["outdir"])
    print(f"Done. Report saved to {path}")

if __name__ == "__main__":
    main()
