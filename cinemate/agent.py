"""CineMate: a conversational front end for the hybrid recommender.

The agent is framework-independent: `reply(text) -> str`. It is used by the console
chat (app.py), the notebook, and the unit tests.

HCI design choices (see report):
  * Natural-language input, not forms ("I loved Inception", "something funny and short").
  * Fuzzy title matching so typos still work.
  * Every recommendation carries a one-line reason ("why?" gives more detail).
  * Proactive behaviour: the agent offers recommendations once it knows enough,
    and onboards new users with quick yes/no ratings of well-known films (cold start).
  * Users stay in control: they can correct ratings, ask for "more", or "reset".
"""
from __future__ import annotations

import difflib
import re

from .data_loader import MovieData
from .models import HybridRecommender

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
          "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance",
          "Sci-Fi", "Thriller", "War", "Western"]
GENRE_ALIASES = {
    "funny": "Comedy", "comedy": "Comedy", "comedies": "Comedy", "scary": "Horror",
    "horror": "Horror", "sci-fi": "Sci-Fi", "scifi": "Sci-Fi", "science fiction": "Sci-Fi",
    "romantic": "Romance", "romance": "Romance", "animated": "Animation", "cartoon": "Animation",
    "kids": "Children", "family": "Children", "action": "Action", "thriller": "Thriller",
    "documentary": "Documentary", "drama": "Drama", "crime": "Crime", "war": "War",
    "western": "Western", "musical": "Musical", "mystery": "Mystery", "fantasy": "Fantasy",
    "adventure": "Adventure", "noir": "Film-Noir",
}
LIKE = r"(?:i\s+)?(?:really\s+)?(?:liked|loved|enjoyed|like|love|enjoy|adored)"
DISLIKE = r"(?:i\s+)?(?:didn'?t|did not|don'?t|do not)\s+(?:like|enjoy)|(?:i\s+)?(?:hated|hate|disliked)"
PROACTIVE_AFTER = 3
ONBOARDING_SIZE = 5


class CineMate:
    def __init__(self, data: MovieData, model: HybridRecommender | None = None):
        self.data = data
        self.model = model or HybridRecommender(data)
        m = data.movies
        self.title_of = dict(zip(m["movieId"], m["clean_title"]))
        self.year_of = dict(zip(m["movieId"], m["year"]))
        votes = data.ratings.groupby("movieId").size()
        # Title index limited to films with some ratings, so fuzzy match finds the famous one.
        self._index = {}
        for mid, t in zip(m["movieId"], m["clean_title"]):
            key = self._norm(t)
            if key not in self._index or votes.get(mid, 0) > votes.get(self._index[key], 0):
                self._index[key] = mid
        self._keys = list(self._index)
        self.onboarding = [int(x) for x in votes[votes > 200].sort_values(ascending=False).index[:12]]
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.liked: dict[int, float] = {}
        self.shown: set[int] = set()
        self.last_recs = []
        self.last_query: dict = {}
        self.onboard_queue: list[int] = []
        self.offered_proactive = False

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _norm(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

    def find_movie(self, text: str) -> int | None:
        key = self._norm(re.sub(r"\(\d{4}\)", "", text))
        if not key:
            return None
        if key in self._index:
            return self._index[key]
        if f"the {key}" in self._index:
            return self._index[f"the {key}"]
        match = difflib.get_close_matches(key, self._keys, n=1, cutoff=0.72)
        return self._index[match[0]] if match else None

    def _label(self, mid: int) -> str:
        y = self.year_of.get(mid)
        return f"{self.title_of[mid]} ({int(y)})" if y == y and y else self.title_of[mid]

    def _format(self, recs, intro: str) -> str:
        if not recs:
            return "I couldn't find anything new that fits. Try telling me about another movie you like."
        self.last_recs = recs
        self.shown |= {r.movie_id for r in recs}
        lines = [intro]
        for n, r in enumerate(recs, 1):
            lines.append(f"  {n}. {self._label(r.movie_id)} - {r.reason}")
        lines.append("Say \"why 2\" for details, \"more\" for others, or tell me what you think of any of them.")
        return "\n".join(lines)

    def _detect_genre(self, text: str) -> str | None:
        t = text.lower()
        for alias, g in sorted(GENRE_ALIASES.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"\b{re.escape(alias)}\b", t):
                return g
        return None

    # ------------------------------------------------------------------ main
    def greet(self) -> str:
        return ("Hi, I'm CineMate. Tell me a few movies you liked (or didn't), and I'll suggest "
                "what to watch next.\nNot sure where to start? Type \"quiz me\" and I'll ask about a "
                "few well-known films.")

    def reply(self, text: str) -> str:
        t = text.strip()
        low = t.lower()
        if not t:
            return "I'm listening. Name a movie you enjoyed, or type \"help\"."

        # Answers during the onboarding quiz (cold start).
        if self.onboard_queue and re.fullmatch(r"(y|yes|yep|loved it|liked it|n|no|nope|skip|not seen|haven'?t seen( it)?|stop)", low):
            return self._onboard_answer(low)

        if low in {"help", "?", "what can you do"}:
            return ("You can say things like:\n"
                    "  - I loved The Matrix / I didn't like Titanic\n"
                    "  - recommend something | something funny | movies like Up\n"
                    "  - why 3 | more | my list | forget Titanic | quiz me | reset | bye")
        if low in {"reset", "start over"}:
            self.reset()
            return "Done. I've forgotten your ratings. " + self.greet().split("\n")[0]
        if low in {"bye", "quit", "exit"}:
            return "Enjoy the movie!"
        if low in {"quiz me", "quiz", "onboard", "i don't know", "not sure"}:
            self.onboard_queue = [m for m in self.onboarding if m not in self.liked][:ONBOARDING_SIZE]
            return self._onboard_next(first=True)
        if low in {"my list", "list", "what do you know about me"}:
            if not self.liked:
                return "You haven't rated anything yet."
            items = [f"  - {self._label(m)}: {'liked' if r >= 3.5 else 'not for you'}" for m, r in self.liked.items()]
            return "Here's what you've told me:\n" + "\n".join(items)

        m = re.match(r"why\s*(?:#|number)?\s*(\d+)", low)
        if m:
            return self._why(int(m.group(1)))

        if low in {"more", "show more", "others", "next"}:
            if not self.last_query:
                return self.recommend()
            return self.recommend(**self.last_query, intro="Here are a few more:")

        m = re.match(r"(?:forget|remove|undo)\s+(.+)", low)
        if m:
            mid = self.find_movie(m.group(1))
            if mid in self.liked:
                self.liked.pop(mid)
                return f"OK, I've removed {self._label(mid)} from your list."
            return "That movie isn't on your list."

        # Rating a numbered item from the last list: "loved 2", "2 was bad"
        m = re.match(r"(liked|loved|good|bad|hated|seen)\s+(\d+)$|(\d+)\s+(?:was\s+)?(good|great|bad|meh)", low)
        if m and self.last_recs:
            n = int(m.group(2) or m.group(3))
            word = m.group(1) or m.group(4)
            if 1 <= n <= len(self.last_recs):
                mid = self.last_recs[n - 1].movie_id
                return self._rate(self.title_of[mid], 1.5 if word in {"bad", "hated", "meh"} else 4.5, mid=mid)

        m = re.match(rf"(?:{DISLIKE})\s+(.+)", low)
        if m:
            return self._rate(m.group(1), 1.5)
        m = re.match(rf"(?:{LIKE})\s+(.+)", low)
        if m and not re.match(r"(movies?|something|films?)\b", m.group(1)):
            return self._rate(m.group(1), 4.5)

        m = re.search(r"(?:movies?|films?|something)\s+(?:like|similar to)\s+(.+)", low)
        if m:
            mid = self.find_movie(m.group(1))
            if mid is None:
                return f"I couldn't find \"{m.group(1)}\". Could you check the title?"
            # Use only the named film as the seed so the answer is about *that* movie.
            recs = self.model.recommend({mid: 5.0}, k=5, exclude=self.shown | set(self.liked))
            return self._format(recs, f"If you liked {self._label(mid)}, try:")

        if re.search(r"\b(recommend|suggest|what should i watch|something|mood|watch)\b", low):
            genre = self._detect_genre(low)
            kw = re.sub(r"\b(recommend|suggest|me|a|an|some|something|movie|movies|film|i'?m|in|the|mood|for|what|should|i|watch|tonight|please)\b", " ", low)
            kw = re.sub(r"\s+", " ", kw).strip() or None
            return self.recommend(genre=genre, keywords=kw)

        # Bare title -> treat as a like, which is the most common thing people type.
        mid = self.find_movie(t)
        if mid is not None:
            return self._rate(t, 4.5, mid=mid)
        return ("Sorry, I didn't catch that. Try \"I loved The Matrix\", \"something funny\", "
                "or type \"help\".")

    # -------------------------------------------------------------- actions
    def recommend(self, genre=None, keywords=None, intro=None) -> str:
        self.last_query = {"genre": genre, "keywords": keywords}
        recs = self.model.recommend(self.liked, k=5, exclude=self.shown, genre=genre, keywords=keywords)
        if intro is None:
            if not self.liked and not keywords:
                intro = "I don't know your taste yet, so here are crowd favorites" + \
                        (f" in {genre}:" if genre else ":")
            else:
                intro = "Based on what you've told me, you might enjoy:"
        return self._format(recs, intro)

    def _rate(self, text, rating, mid=None) -> str:
        mid = mid if mid is not None else self.find_movie(text)
        if mid is None:
            return f"I couldn't find \"{text.strip()}\". Could you check the spelling or add the year?"
        self.liked[mid] = rating
        verb = "liked" if rating >= 3.5 else "didn't like"
        msg = f"Got it, you {verb} {self._label(mid)}."
        n_liked = sum(1 for r in self.liked.values() if r >= 3.5)
        # Proactive step: once there is enough signal, offer recommendations without being asked.
        if n_liked >= PROACTIVE_AFTER and not self.offered_proactive:
            self.offered_proactive = True
            return msg + "\nI think I have a feel for your taste now.\n" + self.recommend()
        if n_liked < PROACTIVE_AFTER:
            msg += f" Tell me {PROACTIVE_AFTER - n_liked} more and I'll start suggesting, or say \"recommend\" now."
        return msg

    def _onboard_next(self, first=False) -> str:
        if not self.onboard_queue:
            return "Thanks! " + self.recommend()
        mid = self.onboard_queue[0]
        prefix = "Quick quiz - answer yes, no, or skip.\n" if first else ""
        return f"{prefix}Did you like {self._label(mid)}?"

    def _onboard_answer(self, low) -> str:
        if low == "stop":
            self.onboard_queue = []
            return "No problem. " + self.recommend()
        mid = self.onboard_queue.pop(0)
        if low.startswith("y") or low.startswith("loved") or low.startswith("liked"):
            self.liked[mid] = 4.5
        elif low in {"n", "no", "nope"}:
            self.liked[mid] = 1.5
        self.offered_proactive = True        # quiz ends in recommendations anyway
        return self._onboard_next()

    def _why(self, n: int) -> str:
        if not self.last_recs or not 1 <= n <= len(self.last_recs):
            return "Ask for recommendations first, then say \"why 1\", \"why 2\", and so on."
        r = self.last_recs[n - 1]
        parts = [f"{self._label(r.movie_id)}: {r.reason}."]
        if r.sources:
            c, g = r.sources["collaborative"], r.sources["content"]
            driver = ("mostly from viewers with tastes like yours" if c >= g
                      else "mostly from genre and tag similarity to films you liked")
            parts.append(f"This match comes {driver}.")
        genres = ", ".join(self.data.movies.iloc[self.model.idx_of[r.movie_id]]["genres"])
        parts.append(f"Genres: {genres}.")
        return " ".join(parts)
