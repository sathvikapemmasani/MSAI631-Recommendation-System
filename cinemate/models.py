"""Recommendation models.

1. PopularityRecommender  - non-personalized baseline (Bayesian-averaged rating).
2. SARRecommender         - item-to-item co-occurrence model, re-implemented from the
                            idea in Microsoft Recommenders' SAR quick-start
                            (https://github.com/recommenders-team/recommenders), the
                            sample this project started from.
3. ContentRecommender     - TF-IDF over genres + user tags (new in this project).
4. HybridRecommender      - weighted blend of SAR + content with cold-start fallback,
                            diversity re-ranking and plain-language explanations
                            (new in this project).

Every model exposes:  recommend(liked: dict[movieId -> rating], k, exclude) -> list[Rec]
so the conversational agent can work with anonymous users who are not in the dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .data_loader import MovieData


@dataclass
class Rec:
    movie_id: int
    score: float
    reason: str = ""
    sources: dict = field(default_factory=dict)


def _top_k(scores: np.ndarray, k: int, banned: set[int]) -> list[int]:
    s = scores.copy()
    if banned:
        s[list(banned)] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    idx = np.argpartition(-s, k - 1)[:k] if k > 0 else np.array([], dtype=int)
    return [int(i) for i in idx[np.argsort(-s[idx])] if np.isfinite(s[i])]


class _Base:
    def __init__(self, data: MovieData):
        self.data = data
        self.movie_ids = data.movies["movieId"].to_numpy()
        self.idx_of = {int(m): i for i, m in enumerate(self.movie_ids)}

    def _banned(self, liked: dict, exclude) -> set[int]:
        banned = {self.idx_of[m] for m in liked if m in self.idx_of}
        banned |= {self.idx_of[m] for m in (exclude or []) if m in self.idx_of}
        return banned


class PopularityRecommender(_Base):
    """Weighted rating (IMDb-style): v/(v+m)*R + m/(v+m)*C."""

    def __init__(self, data: MovieData, ratings: pd.DataFrame | None = None, min_votes: int = 25):
        super().__init__(data)
        r = data.ratings if ratings is None else ratings
        stats = r.groupby("movieId")["rating"].agg(["mean", "count"])
        C = r["rating"].mean()
        m = min_votes
        wr = stats["count"] / (stats["count"] + m) * stats["mean"] + m / (stats["count"] + m) * C
        self.scores = np.full(len(self.movie_ids), -np.inf)
        for mid, val in wr.items():
            if mid in self.idx_of:
                self.scores[self.idx_of[mid]] = val
        self.counts = stats["count"]

    def recommend(self, liked, k=10, exclude=None, genre: str | None = None):
        s = self.scores.copy()
        if genre:
            mask = self.data.movies["genres"].map(lambda g: genre in g).to_numpy()
            s[~mask] = -np.inf
        out = _top_k(s, k, self._banned(liked, exclude))
        return [Rec(int(self.movie_ids[i]), float(s[i]), "popular with MovieLens users") for i in out]


class SARRecommender(_Base):
    """Item similarity from co-occurrence (Jaccard), scored against the user's ratings."""

    def __init__(self, data: MovieData, ratings: pd.DataFrame | None = None, like_threshold: float = 3.5):
        super().__init__(data)
        r = data.ratings if ratings is None else ratings
        r = r[r["rating"] >= like_threshold]
        users = {u: i for i, u in enumerate(r["userId"].unique())}
        rows = r["userId"].map(users).to_numpy()
        cols = r["movieId"].map(self.idx_of).to_numpy()
        A = sparse.csr_matrix((np.ones(len(r)), (rows, cols)), shape=(len(users), len(self.movie_ids)))
        C = (A.T @ A).tocsr().astype(np.float32)          # co-occurrence counts
        diag = C.diagonal()
        C = C.tocoo()
        denom = diag[C.row] + diag[C.col] - C.data
        jac = np.divide(C.data, denom, out=np.zeros_like(C.data), where=denom > 0)
        keep = C.row != C.col
        self.sim = sparse.csr_matrix((jac[keep], (C.row[keep], C.col[keep])), shape=C.shape)

    def recommend(self, liked, k=10, exclude=None):
        if not liked:
            return []
        u = np.zeros(len(self.movie_ids), dtype=np.float32)
        for m, rating in liked.items():
            if m in self.idx_of:
                u[self.idx_of[m]] = rating - 2.5            # centre so dislikes push away
        s = np.asarray(self.sim.T @ u).ravel()
        s[s <= 0] = -np.inf
        out = _top_k(s, k, self._banned(liked, exclude))
        return [Rec(int(self.movie_ids[i]), float(s[i])) for i in out]

    def strongest_link(self, movie_id: int, liked: dict) -> int | None:
        """Which liked movie contributed most to this recommendation (for explanations)."""
        j = self.idx_of[movie_id]
        best, best_val = None, 0.0
        for m, rating in liked.items():
            if m in self.idx_of and rating >= 3.5:
                v = self.sim[self.idx_of[m], j]
                if v > best_val:
                    best, best_val = m, v
        return best


class ContentRecommender(_Base):
    """TF-IDF on genres (weighted x2) + the community's free-text tags."""

    def __init__(self, data: MovieData):
        super().__init__(data)
        tag_text = (data.tags.assign(tag=data.tags["tag"].astype(str).str.lower().str.replace(" ", "_"))
                    .groupby("movieId")["tag"].apply(" ".join))
        docs = []
        for mid, genres in zip(data.movies["movieId"], data.movies["genres"]):
            g = " ".join(x.lower().replace("-", "_") for x in genres)
            docs.append(f"{g} {g} {tag_text.get(mid, '')}")
        self.vectorizer = TfidfVectorizer(token_pattern=r"[a-z0-9_']+", min_df=1)
        self.X = normalize(self.vectorizer.fit_transform(docs))
        self.genre_sets = [set(g) for g in data.movies["genres"]]

    def profile(self, liked: dict):
        vec = None
        for m, rating in liked.items():
            if m in self.idx_of:
                w = rating - 2.5
                row = self.X[self.idx_of[m]] * w
                vec = row if vec is None else vec + row
        return vec

    def recommend(self, liked, k=10, exclude=None, keywords: str | None = None):
        vec = self.profile(liked)
        if keywords:
            kw = self.vectorizer.transform([keywords.lower()])
            vec = kw * 2 if vec is None else vec + kw * 2
        if vec is None:
            return []
        s = np.asarray((self.X @ vec.T).todense()).ravel()
        s[s <= 0] = -np.inf
        out = _top_k(s, k, self._banned(liked, exclude))
        return [Rec(int(self.movie_ids[i]), float(s[i])) for i in out]

    def shared_genres(self, a: int, b: int) -> list[str]:
        return sorted(self.genre_sets[self.idx_of[a]] & self.genre_sets[self.idx_of[b]])


class HybridRecommender(_Base):
    """alpha * SAR + (1-alpha) * content, with popularity prior and MMR-style diversity."""

    def __init__(self, data: MovieData, ratings: pd.DataFrame | None = None, alpha: float = 0.8,
                 diversity: float = 0.15, min_votes: int = 10):
        super().__init__(data)
        self.sar = SARRecommender(data, ratings)
        self.content = ContentRecommender(data)
        self.pop = PopularityRecommender(data, ratings)
        self.alpha, self.diversity, self.min_votes = alpha, diversity, min_votes
        counts = self.pop.counts
        self.votes = np.array([counts.get(m, 0) for m in self.movie_ids])

    @staticmethod
    def _norm(recs):
        if not recs:
            return {}
        v = np.array([r.score for r in recs])
        lo, hi = v.min(), v.max()
        return {r.movie_id: (1.0 if hi == lo else (r.score - lo) / (hi - lo)) for r in recs}

    def recommend(self, liked, k=10, exclude=None, keywords: str | None = None,
                  genre: str | None = None, pool: int = 200, explain: bool = True):
        # Cold start: nothing rated yet -> popularity (optionally within a genre).
        if not liked and not keywords:
            return self.pop.recommend(liked, k, exclude, genre=genre)

        sar = self._norm(self.sar.recommend(liked, pool, exclude)) if liked else {}
        con = self._norm(self.content.recommend(liked, pool, exclude, keywords=keywords))
        a = self.alpha if sar else 0.0
        if keywords:                       # explicit request: lean on content
            a = min(a, 0.3)
        cands = {}
        for mid in set(sar) | set(con):
            i = self.idx_of[mid]
            if self.votes[i] < self.min_votes:              # avoid obscure one-rating films
                continue
            if genre and genre not in self.content.genre_sets[i]:
                continue
            prior = 0.05 * (self.pop.scores[i] - 3.0) if np.isfinite(self.pop.scores[i]) else 0
            cands[mid] = (a * sar.get(mid, 0) + (1 - a) * con.get(mid, 0) + prior,
                          sar.get(mid, 0), con.get(mid, 0))

        # Greedy diversity re-rank (Maximal Marginal Relevance on content vectors).
        chosen: list[int] = []
        if cands:
            ids = list(cands)
            base = np.array([cands[m][0] for m in ids])
            Xc = self.content.X[[self.idx_of[m] for m in ids]]
            S = (Xc @ Xc.T).toarray()
            max_sim = np.zeros(len(ids))
            alive = np.ones(len(ids), dtype=bool)
            for _ in range(min(k, len(ids))):
                val = np.where(alive, base - self.diversity * max_sim, -np.inf)
                j = int(np.argmax(val))
                chosen.append(ids[j])
                alive[j] = False
                max_sim = np.maximum(max_sim, S[j])

        recs = []
        for mid in chosen:
            score, s_sar, s_con = cands[mid]
            recs.append(Rec(mid, float(score), self._explain(mid, liked, keywords) if explain else "",
                            {"collaborative": round(s_sar, 3), "content": round(s_con, 3)}))
        return recs

    def _explain(self, mid: int, liked: dict, keywords: str | None) -> str:
        title = lambda m: self.data.movies.iloc[self.idx_of[m]]["clean_title"]
        anchor = self.sar.strongest_link(mid, liked) if liked else None
        if anchor is not None:
            shared = self.content.shared_genres(mid, anchor)
            extra = f" and shares {', '.join(shared[:2]).lower()}" if shared else ""
            return f"people who liked {title(anchor)} also liked this{extra}"
        if keywords:
            return f"matches what you asked for (\"{keywords}\")"
        if liked:
            best = max(liked, key=lambda m: len(self.content.shared_genres(mid, m)) if m in self.idx_of else -1)
            shared = self.content.shared_genres(mid, best)
            if shared:
                return f"similar {', '.join(shared[:2]).lower()} style to {title(best)}"
        return "a well-rated pick that fits your taste"
