"""Offline evaluation: per-user hold-out of 20% of each user's ratings.

For every test user we give the model only their training ratings, ask for the top 10,
and count how many of their held-out *liked* movies (rating >= 4) appear.
Metrics: Precision@10, Recall@10, catalogue coverage, and intra-list diversity.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .data_loader import MovieData, load
from .models import (ContentRecommender, HybridRecommender, PopularityRecommender,
                     SARRecommender)


def split(ratings: pd.DataFrame, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    test_mask = np.zeros(len(ratings), dtype=bool)
    for _, idx in ratings.groupby("userId").indices.items():
        if len(idx) >= 10:
            n = max(1, int(len(idx) * test_frac))
            test_mask[rng.choice(idx, n, replace=False)] = True
    return ratings[~test_mask], ratings[test_mask]


def evaluate(data: MovieData | None = None, k=10, n_users=200, seed=42) -> pd.DataFrame:
    data = data or load()
    train, test = split(data.ratings, seed=seed)
    models = {
        "Popularity (baseline)": PopularityRecommender(data, train),
        "SAR item-item (original sample)": SARRecommender(data, train),
        "Content TF-IDF (new)": ContentRecommender(data),
        "Hybrid + diversity (final)": HybridRecommender(data, train),
    }
    liked_test = test[test["rating"] >= 4].groupby("userId")["movieId"].apply(set)
    users = liked_test.index.to_numpy()
    rng = np.random.default_rng(seed)
    users = rng.choice(users, min(n_users, len(users)), replace=False)
    train_by_user = {u: dict(zip(g["movieId"], g["rating"])) for u, g in train.groupby("userId")}
    content = models["Content TF-IDF (new)"]

    rows = []
    for name, model in models.items():
        prec, rec, ild, seen = [], [], [], set()
        t0 = time.time()
        for u in users:
            liked = train_by_user.get(u, {})
            kwargs = {"explain": False} if isinstance(model, HybridRecommender) else {}
            recs = [r.movie_id for r in model.recommend(liked, k, **kwargs)]
            hits = len(set(recs) & liked_test[u])
            prec.append(hits / k)
            rec.append(hits / len(liked_test[u]))
            seen |= set(recs)
            if len(recs) > 1:
                X = content.X[[content.idx_of[m] for m in recs]]
                S = (X @ X.T).toarray()
                ild.append(1 - S[np.triu_indices(len(recs), 1)].mean())
        rows.append({"Model": name, f"Precision@{k}": np.mean(prec), f"Recall@{k}": np.mean(rec),
                     "Coverage": len(seen) / len(data.movies), "Diversity": np.mean(ild),
                     "ms/user": 1000 * (time.time() - t0) / len(users)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 120)
    print(evaluate().round(4).to_string(index=False))
