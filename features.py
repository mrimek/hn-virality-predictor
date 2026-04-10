import re
import datetime
import pandas as pd
import numpy as np
from urllib.parse import urlparse

VIRAL_THRESHOLD_PERCENTILE = 90  # top 10% by score = "viral"

_NOW       = datetime.datetime.now(datetime.timezone.utc)
CUTOFF_5Y  = _NOW - datetime.timedelta(days=5 * 365)
CUTOFF_3Y  = _NOW - datetime.timedelta(days=3 * 365)
CUTOFF_1Y  = _NOW - datetime.timedelta(days=365)

SHOW_HN = r"(?i)^show hn"
ASK_HN  = r"(?i)^ask hn"
TELL_HN = r"(?i)^tell hn"

QUESTION_PATTERN   = re.compile(r"\?$")
NUMBER_PATTERN     = re.compile(r"\d+")
PARENS_PATTERN     = re.compile(r"\(([^)]+)\)")
PARENS_HAS_PATTERN = re.compile(r"\([^)]+\)")

# Narrative / authorship
SOLO_BUILD_PATTERN = re.compile(
    r"(?i)^show hn[:\s]+i (built|made|wrote|created|developed|coded|shipped|launched)\b",
)
TEAM_BUILD_PATTERN = re.compile(
    r"(?i)^show hn[:\s]+we (built|made|wrote|created|developed|coded|shipped|launched)\b",
)
BUILT_TIMEFRAME_PATTERN = re.compile(
    r"(?i)\bin\s+\d+\s+(hours?|days?|weeks?|months?|weekends?)\b",
)

# Demo / live product signals (title)
DEMO_PATTERN = re.compile(
    r"(?i)\b(demo|live demo|playground|try it|try now|try online|hosted|web app|webapp)\b",
)

# Pricing signals
FREE_SIGNAL_PATTERN = re.compile(
    r"(?i)\bfree\b|open[- ]source|\$0\b|no[- ]cost|gratis|free[- ]tier|foss",
)
PAID_SIGNAL_PATTERN = re.compile(
    r"(?i)\bpaid\b|\bsubscription\b|\bpricing\b|\$\d+\s*/\s*(mo|month|yr|year)\b",
)

# Individual tech stack
TECH_RUST       = re.compile(r"(?i)\brust\b")
TECH_GO         = re.compile(r"(?i)\bgolang\b|\bgo\b")
TECH_PYTHON     = re.compile(r"(?i)\bpython\b")
TECH_WASM       = re.compile(r"(?i)\bwasm\b|webassembly")
TECH_TYPESCRIPT = re.compile(r"(?i)\btypescript\b")
TECH_CPP        = re.compile(r"(?i)\bc\+\+\b|\bcpp\b")

# Domains that are NOT live-app URLs (used to detect real hosted demos)
_NON_LIVE_DOMAINS = re.compile(
    r"github\.com|gitlab\.com|arxiv\.org|youtube\.com|youtu\.be|"
    r"medium\.com|substack\.com|nytimes\.com|wsj\.com|ft\.com|"
    r"reddit\.com|twitter\.com|x\.com|linkedin\.com|news\.ycombinator\.com",
    re.IGNORECASE,
)

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


def _compute_domain_gap(df: pd.DataFrame) -> pd.Series:
    """
    For each Show HN post, compute days since the same domain last appeared in a Show HN.
    Returns 365 for first occurrence or posts with no domain.
    Non-Show-HN rows get 365.
    """
    result = pd.Series(365.0, index=df.index, dtype=float)

    show_mask = (df["is_show_hn"] == 1) & (df["domain"] != "")
    eligible = df[show_mask].sort_values("dt")
    if eligible.empty or len(eligible) < 2:
        return result

    prev_times = eligible.groupby("domain")["dt"].shift(1)
    deltas = (eligible["dt"] - prev_times).dt.total_seconds() / 86400.0
    deltas = deltas.clip(0, 365).fillna(365)
    result.loc[deltas.index] = deltas
    return result


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input columns: id, title, url, score, time, num_comments, author
    Returns feature columns + score, viral, title, domain, dt.
    """
    df = df.copy()

    # Timestamps
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["hour"]        = df["dt"].dt.hour
    df["day_of_week"] = df["dt"].dt.dayofweek
    df["month"]       = df["dt"].dt.month
    df["year"]        = df["dt"].dt.year

    # Domain (needed early for domain-gap feature)
    df["domain"] = df["url"].apply(extract_domain)

    # Post type flags (needed early for domain-gap)
    df["is_show_hn"] = df["title"].str.contains(SHOW_HN).astype(int)
    df["is_ask_hn"]  = df["title"].str.contains(ASK_HN).astype(int)
    df["is_tell_hn"] = df["title"].str.contains(TELL_HN).astype(int)

    # --- Title features ---
    df["title_len"]        = df["title"].str.len()
    df["title_words"]      = df["title"].str.split().str.len()
    df["title_caps_ratio"] = df["title"].apply(
        lambda t: sum(1 for c in str(t) if c.isupper()) / max(len(str(t)), 1)
    )
    df["is_question"]      = df["title"].str.contains(QUESTION_PATTERN).astype(int)
    df["has_number"]       = df["title"].str.contains(NUMBER_PATTERN).astype(int)
    df["hype_word_count"]  = df["title"].apply(lambda t: len(HYPE_PATTERN.findall(str(t))))
    df["has_parens"]       = df["title"].str.contains(PARENS_HAS_PATTERN).astype(int)
    df["parens_content_len"] = df["title"].apply(
        lambda t: len(m.group(1)) if (m := PARENS_PATTERN.search(str(t))) else 0
    )

    # Title sweet spot: 60–80 chars is the Show HN peak range
    df["title_sweet_spot"] = df["title_len"].between(60, 80).astype(int)

    # Narrative / authorship signals
    df["is_solo_build"]  = df["title"].str.contains(SOLO_BUILD_PATTERN).astype(int)
    df["is_team_build"]  = df["title"].str.contains(TEAM_BUILD_PATTERN).astype(int)
    df["built_timeframe"] = df["title"].str.contains(BUILT_TIMEFRAME_PATTERN).astype(int)

    # Demo / live product signal
    df["has_demo"] = df["title"].str.contains(DEMO_PATTERN).astype(int)

    # GitHub mentioned in title (separate from URL-based is_github)
    df["title_mentions_github"] = df["title"].str.contains(
        r"(?i)github", na=False
    ).astype(int)

    # Pricing signals
    df["has_free_signal"] = df["title"].str.contains(FREE_SIGNAL_PATTERN).astype(int)
    df["has_paid_signal"] = df["title"].str.contains(PAID_SIGNAL_PATTERN).astype(int)

    # Individual tech stack signals
    df["tech_rust"]       = df["title"].str.contains(TECH_RUST).astype(int)
    df["tech_go"]         = df["title"].str.contains(TECH_GO).astype(int)
    df["tech_python"]     = df["title"].str.contains(TECH_PYTHON).astype(int)
    df["tech_wasm"]       = df["title"].str.contains(TECH_WASM).astype(int)
    df["tech_typescript"] = df["title"].str.contains(TECH_TYPESCRIPT).astype(int)
    df["tech_cpp"]        = df["title"].str.contains(TECH_CPP).astype(int)

    # --- Topic features ---
    for topic, pattern in TOPIC_PATTERNS.items():
        df[f"topic_{topic}"] = df["title"].str.contains(pattern).astype(int)
    df["topic_count"] = sum(df[f"topic_{t}"] for t in TOPICS)

    # --- Domain / URL features ---
    df["has_url"]   = (df["url"].notna() & (df["url"] != "")).astype(int)
    df["is_github"] = df["domain"].str.contains("github.com", na=False).astype(int)
    df["is_arxiv"]  = df["domain"].str.contains("arxiv.org", na=False).astype(int)
    df["is_youtube"]= df["domain"].str.contains("youtube.com|youtu.be", na=False).astype(int)
    df["is_medium"] = df["domain"].str.contains("medium.com|substack.com", na=False).astype(int)
    df["is_nytimes"]= df["domain"].str.contains("nytimes.com|wsj.com|ft.com", na=False).astype(int)

    # Live-app URL: has a URL but it's not a code repo, paper, video, or news site
    df["is_live_url"] = (
        (df["has_url"] == 1) &
        (~df["domain"].str.contains(_NON_LIVE_DOMAINS, na=True))
    ).astype(int)

    # Repeat submitter: days since same domain last appeared in Show HN
    df["days_since_domain_show_hn"] = _compute_domain_gap(df)

    # --- Target ---
    threshold = df["score"].quantile(VIRAL_THRESHOLD_PERCENTILE / 100)
    df["viral"] = (df["score"] >= threshold).astype(int)

    return df[FEATURE_COLS + ["score", "viral", "title", "domain", "dt"]]


FEATURE_COLS = [
    # Timing
    "hour", "day_of_week", "month", "year",
    # Title core
    "title_len", "title_words", "title_caps_ratio",
    "is_question", "has_number", "hype_word_count",
    "is_show_hn", "is_ask_hn", "is_tell_hn",
    "has_parens", "parens_content_len",
    # Title signals (new)
    "title_sweet_spot",
    "is_solo_build", "is_team_build", "built_timeframe",
    "has_demo", "title_mentions_github",
    "has_free_signal", "has_paid_signal",
    # Tech stack (new)
    "tech_rust", "tech_go", "tech_python", "tech_wasm", "tech_typescript", "tech_cpp",
    # Domain / URL
    "has_url", "is_github", "is_arxiv", "is_youtube", "is_medium", "is_nytimes",
    "is_live_url",                    # new
    "days_since_domain_show_hn",      # new
    # Topic taxonomy
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
