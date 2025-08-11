# app_chat.py — clean, cloud-safe chat UI
import os, re
import pandas as pd
import streamlit as st
from datetime import datetime

from reddit_coffee_pulse import (
    load_config, init_reddit, pull_posts, sentiment_scores
)

@st.cache_data(ttl=600)
def cached_pull(subreddits, keywords, days, max_posts):
    reddit = init_reddit()
    return pull_posts(reddit, subreddits, keywords, days, max_posts)

TIME_RULES = [
    (r"\btoday\b", 1),
    (r"\byesterday\b", 2),
    (r"\bthis week\b|\bpast week\b", 7),
    (r"\blast week\b", 14),
    (r"\bthis month\b", 30),
    (r"\blast month\b", 60),
]
COMPLAINT_WORDS = {
    "broken","break","defect","leak","leaking","refund","return","warranty",
    "bad","awful","trash","cheap","plastic","waste","inconsistent","weak","bitter",
    "noisy","clog","clogged","stopped","stop","doesn't work","doesnt work","won't","wont",
    "disappointed","disappointing","issue","problem","complain","complaint","customer service"
}

def guess_days(q: str, fallback: int):
    ql = q.lower()
    for pat, d in TIME_RULES:
        if re.search(pat, ql):
            return d
    return fallback

def extract_keywords(q: str):
    if "," in q:
        return [t.strip() for t in q.split(",") if t.strip()]
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-']+", q.lower())
    drop = {"who","what","tell","me","about","people","this","that","those","and","or",
            "the","in","on","week","month","today","yesterday","last","complained","complain"}
    out = [t for t in toks if len(t) >= 3 and t not in drop]
    return out or ["coffee"]

def is_complaint(text_lower: str):
    return any(w in text_lower for w in COMPLAINT_WORDS)

st.set_page_config(page_title="Reddit Q&A", page_icon="🔎", layout="wide")
st.title("🔎 Reddit Q&A")

cfg = load_config("config.yaml")
default_subs = cfg.get("subreddits", ["coffee","espresso","barista","sustainability","zerowaste"])
default_days = int(cfg.get("since_days", 7))
default_max  = int(cfg.get("max_posts", 300))

with st.sidebar:
    st.subheader("Settings")
    subs = st.text_input("Subreddits (comma-separated)", ", ".join(default_subs))
    days = st.slider("Days back", 1, 30, min(default_days, 14))
    max_posts = st.slider("Max posts", 50, 500, min(default_max, 100), step=50)
    focus_complaints = st.checkbox("Focus on complaints", True)
    show_positives = st.checkbox("Include strong positives", False)

if "history" not in st.session_state:
    st.session_state.history = []
for role, msg in st.session_state.history:
    with st.chat_message(role):
        st.markdown(msg)

prompt = st.chat_input("Ask (e.g., 'who complained about keurig this week?')")

if prompt:
    st.session_state.history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    keywords = extract_keywords(prompt)
    days_eff = guess_days(prompt, days)
    subreddits = [s.strip() for s in subs.split(",") if s.strip()]

    with st.chat_message("assistant"):
        with st.status("Fetching and filtering posts…", expanded=False):
            df = cached_pull(subreddits, keywords, days_eff, max_posts)
            if df.empty:
                st.write("_No posts fetched. Try more days, simpler keywords, or more subreddits._")
                st.stop()

            df["sentiment"] = sentiment_scores(df["text"].tolist())
            tl = df["text"].str.lower()

            brand_regex = re.compile("|".join(re.escape(k) for k in keywords), re.I)
            mask_brand = tl.str.contains(brand_regex)

            if focus_complaints:
                mask_compl = (df["sentiment"] == -1) | tl.apply(is_complaint)
            else:
                mask_compl = pd.Series([True] * len(df))

            if not show_positives and focus_complaints:
                mask = mask_brand & mask_compl & ~(df["sentiment"] == 1)
            else:
                mask = mask_brand & mask_compl

            relevant = df[mask].copy()
            if relevant.empty:
                st.write("_No matching posts for that request._")
                st.stop()

            if "url" not in relevant.columns:
                relevant["url"] = "https://www.reddit.com/comments/" + relevant["id"].astype(str)

            relevant["date"] = pd.to_datetime(relevant["date"])
            relevant = relevant.sort_values("date")
            relevant["recency_rank"] = relevant["date"].rank(pct=True)
            relevant["eng_rank"] = (relevant["score"].rank(pct=True) + relevant["num_comments"].rank(pct=True)) / 2
            relevant["rank"] = 0.6 * relevant["recency_rank"] + 0.4 * relevant["eng_rank"]
            relevant = relevant.sort_values("rank", ascending=False)

        total = len(relevant)
        pos = int((relevant["sentiment"] == 1).sum())
        neu = int((relevant["sentiment"] == 0).sum())
        neg = int((relevant["sentiment"] == -1).sum())
        summary = f"{total} relevant posts. Sentiment: {round(100*pos/total)}% positive, {round(100*neu/total)}% neutral, {round(100*neg/total)}% negative."

        st.markdown(f"**Answer:** {summary}")

        def fmt_row(r):
            date_str = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
            senti = {1:"🙂",0:"😐",-1:"☹️"}.get(int(r["sentiment"]),"?")
            title = (r.get("title","") or "").strip() or r["text"][:120].strip()
            return f"- {date_str} r/{r['subreddit']} {senti} — [{title}]({r['url']})"

        topn = relevant.head(15)
        bullets = "\n".join(fmt_row(r) for _, r in topn.iterrows())
        st.markdown("**Top matches:**")
        st.markdown(bullets)

        st.markdown("**Full table:**")
        show_cols = ["date","subreddit","title","query","sentiment","url"]
        for c in show_cols:
            if c not in relevant.columns:
                relevant[c] = "" if c != "sentiment" else 0
        st.dataframe(relevant[show_cols], use_container_width=True)

        os.makedirs("answers", exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        outfile = f"answers/brief_{stamp}.md"
        with open(outfile, "w", encoding="utf-8") as f:
            f.write(f"# Answer\n\nPrompt: {prompt}\n\n{summary}\n\n## Top matches\n{bullets}\n")
        with open(outfile, "rb") as f:
            st.download_button("⬇️ Download brief (Markdown)", data=f.read(),
                               file_name=os.path.basename(outfile), mime="text/markdown")

        st.session_state.history.append(("assistant", f"**Answer:** {summary}\n\n**Top matches:**\n{bullets}"))
