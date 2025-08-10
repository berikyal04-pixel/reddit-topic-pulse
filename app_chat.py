# ===== EMERGENCY PAUSE SWITCH =====
PAUSE_APP = True
# ==================================

# app_chat.py — Text prompt → relevant Reddit posts (uses your existing backend)
import re, os
import pandas as pd
import streamlit as st
from datetime import datetime

from reddit_coffee_pulse import (
    load_config, init_reddit, pull_posts, sentiment_scores, cluster_texts
)

st.set_page_config(page_title="Reddit Q&A", page_icon="🔎", layout="wide")
st.title("🔎 Reddit Q&A")

# --- Settings (from config) ---
cfg = load_config("config.yaml")
default_subs = cfg.get("subreddits", ["coffee","barista","espresso","sustainability","zerowaste"])
default_days = int(cfg.get("since_days", 7))
default_max = int(cfg.get("max_posts", 300))
default_k   = int(cfg.get("k_clusters", 6))

with st.sidebar:
    st.subheader("Advanced")
    subs = st.text_input("Subreddits (comma-separated)", ", ".join(default_subs))
    days = st.slider("Days back", 1, 30, default_days)
    max_posts = st.slider("Max posts", 50, 1000, default_max, step=50)
    k = st.slider("Themes (clusters)", 2, 10, min(default_k, 6))
    focus_complaints = st.checkbox("Focus on complaints", True)

prompt = st.chat_input("Ask (e.g., 'who complained about keurig this week?')")
if 'PAUSE_APP' in globals() and PAUSE_APP:
    st.warning("App is temporarily paused. No Reddit requests will be made.")
    st.stop()

# Show previous chat
if "history" not in st.session_state:
    st.session_state.history = []
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

def parse_keywords(q: str):
    # split by commas if present, else keep words ≥3 chars (basic)
    if "," in q:
        return [t.strip() for t in q.split(",") if t.strip()]
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-']+", q.lower())
    drop = {"who","what","tell","me","about","people","this","that","those","and","or",
            "the","in","on","week","month","today","yesterday","last","complained","complain"}
    out = [t for t in toks if len(t) >= 3 and t not in drop]
    # ensure at least one term
    return out or ["coffee"]

TIME_RULES = [
    (r"\btoday\b", 1),
    (r"\byesterday\b", 2),
    (r"\bthis week\b|\bpast week\b", 7),
    (r"\blast week\b", 14),
    (r"\bthis month\b", 30),
    (r"\blast month\b", 60),
]

def guess_days(q: str, fallback: int):
    ql = q.lower()
    for pat, d in TIME_RULES:
        if re.search(pat, ql):
            return d
    return fallback

COMPLAINT_WORDS = {
    "broken","break","defect","leak","leaking","refund","return","warranty",
    "bad","awful","trash","cheap","plastic","waste","inconsistent","weak","bitter",
    "noisy","clog","clogged","stopped","stop","doesn’t work","doesnt work","won’t","wont",
    "disappointed","disappointing","issue","problem","complain","complaint","customer service"
}

def is_complaint(text_lower: str):
    return any(w in text_lower for w in COMPLAINT_WORDS)

if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    keywords = parse_keywords(prompt)
    days_eff = guess_days(prompt, days)
    subreddits = [s.strip() for s in subs.split(",") if s.strip()]

    with st.chat_message("assistant"):
        with st.status("Fetching posts…", expanded=False):
            try:
                reddit = init_reddit()
            except Exception as e:
                st.error(f"Reddit auth failed. Check secrets/.env. Details: {e}")
                st.stop()

            df = pull_posts(
                reddit,
                subreddits=subreddits,
                queries=keywords,
                since_days=days_eff,
                max_posts=max_posts
            )

            if df.empty:
                st.write("_No posts fetched. Try more days, more subreddits, or simpler keywords._")
                st.stop()

            df["sentiment"] = sentiment_scores(df["text"].tolist())
            tl = df["text"].str.lower()

            # Require at least one keyword present
            brand_regex = re.compile("|".join(re.escape(k) for k in keywords), re.I)
            mask_brand = tl.str.contains(brand_regex)

            # Focus on complaints (negative sentiment OR complaint words)
            if focus_complaints:
                mask_compl = (df["sentiment"] == -1) | tl.apply(is_complaint)
            else:
                mask_compl = pd.Series([True]*len(df))

            relevant = df[mask_brand & mask_compl].copy()
            if relevant.empty:
                st.write("_No matching posts for that request._")
                st.stop()

            # Build URL if missing
            if "url" not in relevant.columns:
                relevant["url"] = "https://www.reddit.com/comments/" + relevant["id"].astype(str)

            # Rank: recency + engagement
            relevant["date"] = pd.to_datetime(relevant["date"])
            relevant = relevant.sort_values("date")
            relevant["recency_rank"] = relevant["date"].rank(pct=True)
            relevant["eng_rank"] = (relevant["score"].rank(pct=True) + relevant["num_comments"].rank(pct=True))/2
            relevant["rank"] = 0.6*relevant["recency_rank"] + 0.4*relevant["eng_rank"]
            relevant = relevant.sort_values("rank", ascending=False)

        # Compose answer
        total = len(relevant)
        pos = int((relevant["sentiment"] == 1).sum())
        neu = int((relevant["sentiment"] == 0).sum())
        neg = int((relevant["sentiment"] == -1).sum())
        summary = f"{total} relevant posts. Sentiment: {round(100*pos/total)}% positive, {round(100*neu/total)}% neutral, {round(100*neg/total)}% negative."

        st.markdown(f"**Answer:** {summary}")
        topn = relevant.head(15).copy()

        def fmt_row(r):
            date_str = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
            senti = {1:"🙂",0:"😐",-1:"☹️"}.get(int(r["sentiment"]),"?")
            title = (r.get("title","") or "").strip()
            title = title if title else r["text"][:120].strip()
            return f"- {date_str} r/{r['subreddit']} {senti} — [{title}]({r['url']})"

        bullets = "\n".join(fmt_row(r) for _, r in topn.iterrows())
        st.markdown("**Top matches:**")
        st.markdown(bullets)

        # Show table
        st.markdown("**Full table (filterable):**")
        show_cols = ["date","subreddit","title","query","sentiment","url"]
        for c in show_cols:
            if c not in topn.columns:
                relevant[c] = "" if c != "sentiment" else 0
        st.dataframe(relevant[show_cols], use_container_width=True)

        # Download brief
        os.makedirs("answers", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        outfile = f"answers/brief_{stamp}.md"
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(f"# Answer\n\nPrompt: {prompt}\n\n{summary}\n\n## Top matches\n{bullets}\n")
        with open(outfile, "rb") as f:
            st.download_button("⬇️ Download brief (Markdown)", data=f.read(),
                               file_name=os.path.basename(outfile), mime="text/markdown")

        st.session_state.history.append(("assistant", f"**Answer:** {summary}\n\n**Top matches:**\n{bullets}"))
