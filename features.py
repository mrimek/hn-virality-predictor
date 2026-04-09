import re
import datetime
import pandas as pd
import numpy as np
from urllib.parse import urlparse

VIRAL_THRESHOLD_PERCENTILE = 90  # top 10% by score = "viral"

_NOW       = datetime.datetime.now(datetime.timezone.utc)
CUTOFF_5Y  = _NOW - datetime.timedelta(days=5 * 365)
CUTOFF_1Y  = _NOW - datetime.timedelta(days=365)

SHOW_HN = r"(?i)^show hn"
ASK_HN  = r"(?i)^ask hn"
TELL_HN = r"(?i)^tell hn"

QUESTION_PATTERN   = re.compile(r"\?$")
NUMBER_PATTERN     = re.compile(r"\d+")
PARENS_PATTERN     = re.compile(r"\(([^)]+)\)")
PARENS_HAS_PATTERN = re.compile(r"\([^)]+\)")

# Topic taxonomy — keyword patterns per category
TOPICS = {
    "ai_ml":     r"gpt|llm|ai\b|ml\b|neural|openai|anthropic|gemini|claude|chatgpt|"
                 r"machine.?learning|deep.?learning|diffusion|transformer|embedding|"
                 r"copilot|stable.?diffusion|midjourney|dall.?e|hugging.?face",
    "security":  r"hack|breach|vulnerab|exploit|malware|ransomware|phishing|"
                 r"zero.?day|cve|backdoor|surveillance|spyware|leak|password|"
                 r"cybersecurity|infosec|fbi|nsa|cia",
    "systems":   r"\brust\b|golang|\bgo\b|c\+\+|linux|kernel|compiler|wasm|"
                 r"performance|latency|throughput|memory|cpu|concurren|async|"
                 r"distributed|database|postgres|sqlite|redis|kafka|clickhouse",
    "web_dev":   r"javascript|typescript|react|vue|angular|svelte|next\.?js|"
                 r"css|html|frontend|backend|rest.?api|graphql|node\.?js|deno|bun",
    "hardware":  r"\bchip\b|gpu|apple.?silicon|m[1-4]\b|nvidia|amd|arm\b|risc|"
                 r"fpga|semiconductor|transistor|fabrication|tsmc|intel",
    "science":   r"research|study|cancer|climate|physics|biology|chemistry|"
                 r"quantum|genome|crispr|protein|spacex|nasa|telescope|satellite",
    "business":  r"startup|funding|raises|\$[0-9]+[bm]\b|ipo|acqui|layoff|"
                 r"valuation|revenue|profit|bankrupt|unicorn|series.[a-e]",
    "policy":    r"law|ban|regulat|congress|government|court|gdpr|antitrust|"
                 r"privacy|censorship|lawsuit|patent|copyright|dmca|election",
    "open_src":  r"open.?source|open.?core|mit.license|apache.license|"
                 r"self.?host|foss|libre|github\.com",
}
TOPIC_PATTERNS = {k: re.compile(v, re.IGNORECASE) for k, v in TOPICS.items()}

HYPE_PATTERN = re.compile(
    r"gpt|llm|\bai\b|rust|wasm|open.?source|launch|startup|raises|funding|"
    r"acqui|ipo|dead|killed|ban|leak|hack|breach|fbi|lawsuit|cancer|cure|"
    r"breakthrough|faster|cheaper|\bfree\b",
    re.IGNORECASE,
)


def extract_domain(url):
    if not url or not isinstance(url, str):
        return ""
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return ""


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns: id, title, url, score, time, num_comments, author
    Returns feature columns + score, viral, title, domain.
    """
    df = df.copy()

    # Timestamps
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["hour"]        = df["dt"].dt.hour
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["month"]       = df["dt"].dt.month
    df["year"]        = df["dt"].dt.year

    # Title features
    df["title_len"]        = df["title"].str.len()
    df["title_words"]      = df["title"].str.split().str.len()
    df["title_caps_ratio"] = df["title"].apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
    )
    df["is_question"]      = df["title"].str.contains(QUESTION_PATTERN).astype(int)
    df["has_number"]       = df["title"].str.contains(NUMBER_PATTERN).astype(int)
    df["hype_word_count"]  = df["title"].apply(lambda t: len(HYPE_PATTERN.findall(str(t))))
    df["is_show_hn"]       = df["title"].str.contains(SHOW_HN).astype(int)
    df["is_ask_hn"]        = df["title"].str.contains(ASK_HN).astype(int)
    df["is_tell_hn"]       = df["title"].str.contains(TELL_HN).astype(int)
    df["has_parens"]       = df["title"].str.contains(PARENS_HAS_PATTERN).astype(int)
    df["parens_content_len"] = df["title"].apply(
        lambda t: len(m.group(1)) if (m := PARENS_PATTERN.search(str(t))) else 0
    )

    # Topic features (one per category)
    for topic, pattern in TOPIC_PATTERNS.items():
        df[f"topic_{topic}"] = df["title"].str.contains(pattern).astype(int)
    df["topic_count"] = sum(df[f"topic_{t}"] for t in TOPICS)

    # Domain / URL features
    df["domain"]    = df["url"].apply(extract_domain)
    df["has_url"]   = (df["url"].notna() & (df["url"] != "")).astype(int)
    df["is_github"] = df["domain"].str.contains("github.com", na=False).astype(int)
    df["is_arxiv"]  = df["domain"].str.contains("arxiv.org", na=False).astype(int)
    df["is_youtube"]= df["domain"].str.contains("youtube.com|youtu.be", na=False).astype(int)
    df["is_medium"] = df["domain"].str.contains("medium.com|substack.com", na=False).astype(int)
    df["is_nytimes"]= df["domain"].str.contains("nytimes.com|wsj.com|ft.com", na=False).astype(int)

    # Target (computed per-subset so threshold is always relative to training data)
    threshold = df["score"].quantile(VIRAL_THRESHOLD_PERCENTILE / 100)
    df["viral"] = (df["score"] >= threshold).astype(int)

    return df[FEATURE_COLS + ["score", "viral", "title", "domain", "dt"]]


FEATURE_COLS = [
    "hour", "day_of_week", "month", "year",
    "title_len", "title_words", "title_caps_ratio",
    "is_question", "has_number", "hype_word_count",
    "is_show_hn", "is_ask_hn", "is_tell_hn",
    "has_parens", "parens_content_len",
    "has_url", "is_github", "is_arxiv", "is_youtube", "is_medium", "is_nytimes",
    # topic features
    "topic_ai_ml", "topic_security", "topic_systems", "topic_web_dev",
    "topic_hardware", "topic_science", "topic_business", "topic_policy",
    "topic_open_src", "topic_count",
]


def features_for_prediction(title: str, url: str = "", posted_at=None) -> dict:
    """Build a single-row feature dict for inference (no score/viral needed)."""
    if posted_at is None:
        posted_at = datetime.datetime.now(datetime.timezone.utc)

    row = pd.DataFrame([{
        "title": title, "url": url, "score": 0,
        "time": int(posted_at.timestamp()),
        "num_comments": 0, "author": "",
    }])
    feats = extract_features(row)
    return feats[FEATURE_COLS].iloc[0].to_dict()
