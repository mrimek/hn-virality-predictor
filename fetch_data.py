import duckdb
from pathlib import Path

Path("data").mkdir(exist_ok=True)
OUTPUT = "data/hn_stories.parquet"

print("Connecting to Hugging Face dataset via DuckDB...")
print("Fetching stories only — this takes 3-10 minutes.\n")

conn = duckdb.connect()
conn.execute("INSTALL httpfs; LOAD httpfs;")

conn.execute(f"""
COPY (
    SELECT id, title, url, score, time, descendants AS num_comments, "by" AS author
    FROM read_parquet('hf://datasets/open-index/hacker-news/data/*/*.parquet')
    WHERE type = 1 AND title != '' AND deleted = 0 AND dead = 0 AND score > 0
)
TO '{OUTPUT}' (FORMAT PARQUET, COMPRESSION ZSTD)
""")

result = conn.execute(f"SELECT count(*), max(score) FROM read_parquet('{OUTPUT}')").fetchone()
print(f"\n✓ Done! {result[0]:,} stories saved to {OUTPUT} (max score: {result[1]})")
print("Next: python3 train.py --data data/hn_stories.parquet")
