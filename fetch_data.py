import time
import duckdb
import pandas as pd
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download

REPO = "open-index/hacker-news"
OUTPUT = "data/hn_stories.parquet"
CACHE = Path("data/raw")
CACHE.mkdir(parents=True, exist_ok=True)

# Fetch list of parquet files
print("Listing dataset files...")
files = sorted(
    f for f in list_repo_files(REPO, repo_type="dataset")
    if f.endswith(".parquet")
)
print(f"Found {len(files)} monthly parquet files\n")

# Download each file with retry
def download_with_retry(filename, retries=5):
    for attempt in range(retries):
        try:
            path = hf_hub_download(
                repo_id=REPO,
                filename=filename,
                repo_type="dataset",
                local_dir=str(CACHE),
            )
            return path
        except Exception as e:
            if attempt < retries - 1:
                wait = 2 ** attempt * 5
                print(f"  Retry {attempt+1}/{retries} after {wait}s ({e})")
                time.sleep(wait)
            else:
                print(f"  Skipping {filename} after {retries} failures: {e}")
                return None

conn = duckdb.connect()
conn.execute("INSTALL httpfs; LOAD httpfs;")

tables = []
for i, filename in enumerate(files, 1):
    dest = CACHE / filename
    if dest.exists():
        print(f"[{i}/{len(files)}] Cached: {filename}")
    else:
        print(f"[{i}/{len(files)}] Downloading: {filename}")
        path = download_with_retry(filename)
        if path is None:
            continue
        time.sleep(0.5)  # be polite to HF

    try:
        query = (
            f"SELECT id, title, url, score, epoch(time) AS time, "
            f"descendants AS num_comments, \"by\" AS author "
            f"FROM read_parquet('{dest}') "
            f"WHERE type = 1 AND title != '' AND deleted = 0 AND dead = 0 AND score > 0"
        )
        df = conn.execute(query).fetchdf()
        tables.append(df)
    except Exception as e:
        print(f"  Error reading {filename}: {e}")

print(f"\nMerging {len(tables)} tables...")
combined = pd.concat(tables, ignore_index=True)
combined.to_parquet(OUTPUT, compression="zstd", index=False)

print(f"Done! {len(combined):,} stories saved to {OUTPUT}")
print(f"Next: python3 train.py")
