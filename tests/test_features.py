import pytest
import pandas as pd
import sys
sys.path.insert(0, ".")

from features import extract_features, features_for_prediction


SAMPLE = pd.DataFrame([
    {"id": 1, "title": "Show HN: My fast Rust HTTP server", "url": "https://github.com/foo/bar",
     "score": 500, "time": 1700000000, "num_comments": 120, "author": "alice"},
    {"id": 2, "title": "Ask HN: What is the best Python book?", "url": "",
     "score": 10, "time": 1700003600, "num_comments": 5, "author": "bob"},
    {"id": 3, "title": "OpenAI raises $10B funding round", "url": "https://nytimes.com/article",
     "score": 800, "time": 1700007200, "num_comments": 300, "author": "carol"},
])


def test_extract_features_shape():
    df = extract_features(SAMPLE)
    assert len(df) == 3


def test_viral_column_exists():
    df = extract_features(SAMPLE)
    assert "viral" in df.columns
    assert df["viral"].isin([0, 1]).all()


def test_show_hn_flag():
    df = extract_features(SAMPLE)
    assert df.iloc[0]["is_show_hn"] == 1
    assert df.iloc[1]["is_show_hn"] == 0


def test_ask_hn_flag():
    df = extract_features(SAMPLE)
    assert df.iloc[1]["is_ask_hn"] == 1
    assert df.iloc[0]["is_ask_hn"] == 0


def test_is_question():
    df = extract_features(SAMPLE)
    assert df.iloc[1]["is_question"] == 1
    assert df.iloc[0]["is_question"] == 0


def test_has_number():
    df = extract_features(SAMPLE)
    assert df.iloc[2]["has_number"] == 1  # "10" in "$10B" contains digits


def test_is_github():
    df = extract_features(SAMPLE)
    assert df.iloc[0]["is_github"] == 1
    assert df.iloc[1]["is_github"] == 0


def test_is_nytimes():
    df = extract_features(SAMPLE)
    assert df.iloc[2]["is_nytimes"] == 1


def test_has_url():
    df = extract_features(SAMPLE)
    assert df.iloc[0]["has_url"] == 1
    assert df.iloc[1]["has_url"] == 0


def test_features_for_prediction_returns_dict():
    feats = features_for_prediction("Show HN: test", "https://github.com/x/y")
    assert isinstance(feats, dict)
    assert "is_show_hn" in feats
    assert feats["is_show_hn"] == 1


def test_features_for_prediction_no_url():
    feats = features_for_prediction("Ask HN: something?")
    assert feats["has_url"] == 0
    assert feats["is_ask_hn"] == 1
    assert feats["is_question"] == 1
