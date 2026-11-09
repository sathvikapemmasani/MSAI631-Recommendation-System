"""Loads the MovieLens "latest-small" dataset and prepares it for the recommenders.

Data source: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM TiiS 5, 4, Article 19. https://grouplens.org/datasets/movielens/
The CSV files are NOT redistributed in this repository; run tools/download_data.py first.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


@dataclass
class MovieData:
    movies: pd.DataFrame   # movieId, title, clean_title, year, genres (list)
    ratings: pd.DataFrame  # userId, movieId, rating, timestamp
    tags: pd.DataFrame     # userId, movieId, tag, timestamp


def _clean_title(title: str) -> str:
    """'Matrix, The (1999)' -> 'The Matrix'."""
    t = YEAR_RE.sub("", title).strip()
    m = re.match(r"^(.*), (The|A|An)$", t)
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    return t


def load(data_dir: str = DATA_DIR) -> MovieData:
    path = lambda f: os.path.join(data_dir, f)
    if not os.path.exists(path("movies.csv")):
        raise FileNotFoundError(
            f"MovieLens files not found in {data_dir}. Run: python tools/download_data.py"
        )
    movies = pd.read_csv(path("movies.csv"))
    ratings = pd.read_csv(path("ratings.csv"))
    tags = pd.read_csv(path("tags.csv")) if os.path.exists(path("tags.csv")) else \
        pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])

    movies["year"] = movies["title"].str.extract(YEAR_RE)[0].astype("float")
    movies["clean_title"] = movies["title"].map(_clean_title)
    movies["genres"] = movies["genres"].map(
        lambda g: [] if g == "(no genres listed)" else g.split("|")
    )
    return MovieData(movies=movies, ratings=ratings, tags=tags)
