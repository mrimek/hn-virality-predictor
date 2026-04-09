"""
Analyze what topics and patterns correlate with virality.

Usage:
    python3 analyze.py                        # full dataset
    python3 analyze.py --subset recent_1y     # past year only
    python3 analyze.py --subset show_hn       # Show HN posts only
    python3 analyze.py --subset recent_5y     # past 5 years
"""
import argparse
import pandas as pd
from features import extract_features, TOPICS, CUTOFF_1Y, CUTOFF_5Y

DATA = "data/hn_stories.parquet"

SUBSETS = {
    "full":      lambda df: df,
    "show_hn":   lambda df: df[df["is_show_hn"] == 1],
    "recent_5y": lambda df: df[df["dt"] >= CUTOFF_5Y],
    "recent_1y": lambda df: df[df["dt"] >= CUTOFF_1Y],
}


def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")


def topic_virality(df: pd.DataFrame):
    section("Virality rate by topic")
    rows = []
    for topic in TOPICS:
        col = f"topic_{topic}"
        sub = df[df[col] == 1]
        if len(sub) < 50:
            continue
        rows.append({
            "topic": topic,
            "posts": len(sub),
            "viral_%": round(sub["viral"].mean() * 100, 1),
            "median_score": int(sub["score"].median()),
            "p90_score": int(sub["score"].quantile(0.9)),
        })
    result = pd.DataFrame(rows).sort_values("viral_%", ascending=False)
    print(result.to_string(index=False))


def keyword_virality(df: pd.DataFrame, top_n=30):
    section(f"Top {top_n} title keywords by virality rate (min 100 posts)")
    import re
    stopwords = {
        "the", "and", "for", "with", "this", "that", "are", "was", "has",
        "have", "from", "its", "not", "but", "you", "your", "can", "new",
        "how", "why", "what", "when", "will", "more", "all", "one", "now",
        "get", "use", "our", "show", "ask", "tell", "using", "into", "out",
    }

    # Explode titles into (word, viral) pairs — vectorized
    df2 = df[["title", "viral"]].copy()
    df2["words"] = df2["title"].str.lower().str.findall(r"\b[a-zA-Z]{3,}\b")
    exploded = df2.explode("words").rename(columns={"words": "keyword"})
    exploded = exploded[~exploded["keyword"].isin(stopwords)]

    agg = (
        exploded.groupby("keyword")["viral"]
        .agg(posts="count", viral_sum="sum")
        .query("posts >= 100")
    )
    agg["viral_%"] = (agg["viral_sum"] / agg["posts"] * 100).round(1)
    result = agg.sort_values("viral_%", ascending=False).head(top_n)[["posts", "viral_%"]]
    print(result.to_string())


def temporal_trend(df: pd.DataFrame):
    section("Topic virality rate by year (top topics)")
    top_topics = [f"topic_{t}" for t in TOPICS]
    rows = []
    for year, grp in df.groupby("year"):
        if len(grp) < 200:
            continue
        row = {"year": year, "posts": len(grp), "overall_viral_%": round(grp["viral"].mean() * 100, 1)}
        for col in top_topics:
            sub = grp[grp[col] == 1]
            if len(sub) >= 20:
                row[col.replace("topic_", "")] = round(sub["viral"].mean() * 100, 1)
            else:
                row[col.replace("topic_", "")] = None
        rows.append(row)
    result = pd.DataFrame(rows).set_index("year")
    topic_cols = [t for t in TOPICS if t in result.columns]
    print(result[["posts", "overall_viral_%"] + topic_cols].to_string())


def posting_time(df: pd.DataFrame):
    section("Best posting hours (UTC) by avg virality rate")
    by_hour = (
        df.groupby("hour")["viral"]
        .agg(posts="count", viral_rate=lambda x: round(x.mean() * 100, 1))
        .sort_values("viral_rate", ascending=False)
    )
    print(by_hour.head(10).to_string())

    section("Best posting days by avg virality rate")
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_day = (
        df.groupby("day_of_week")["viral"]
        .agg(posts="count", viral_rate=lambda x: round(x.mean() * 100, 1))
    )
    by_day.index = [days[i] for i in by_day.index]
    print(by_day.sort_values("viral_rate", ascending=False).to_string())


def main(data_path: str, subset: str):
    print(f"Loading {data_path}...")
    raw = pd.read_parquet(data_path)
    print("Extracting features...")
    df = extract_features(raw)
    df = SUBSETS[subset](df)
    print(f"Analyzing subset '{subset}': {len(df):,} posts\n")

    topic_virality(df)
    keyword_virality(df)
    temporal_trend(df)
    posting_time(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--subset", default="full", choices=list(SUBSETS.keys()))
    args = parser.parse_args()
    main(args.data, args.subset)
