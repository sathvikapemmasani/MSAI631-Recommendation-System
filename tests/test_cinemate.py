"""Unit tests. Run with:  python -m pytest -q   (or python -m unittest discover tests)"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cinemate.agent import CineMate  # noqa: E402
from cinemate.data_loader import _clean_title, load  # noqa: E402
from cinemate.evaluate import split  # noqa: E402
from cinemate.models import (ContentRecommender, HybridRecommender,  # noqa: E402
                             PopularityRecommender, SARRecommender)

DATA = load()
HYBRID = HybridRecommender(DATA)
MATRIX, TOY_STORY, TITANIC = 2571, 1, 1721


class DataTests(unittest.TestCase):
    def test_loaded(self):
        self.assertGreater(len(DATA.movies), 9000)
        self.assertGreater(len(DATA.ratings), 100000)

    def test_clean_title(self):
        self.assertEqual(_clean_title("Matrix, The (1999)"), "The Matrix")
        self.assertEqual(_clean_title("Toy Story (1995)"), "Toy Story")

    def test_split_keeps_every_user_in_train(self):
        train, test = split(DATA.ratings)
        self.assertEqual(set(train.userId), set(DATA.ratings.userId))
        self.assertAlmostEqual(len(test) / len(DATA.ratings), 0.2, delta=0.02)


class ModelTests(unittest.TestCase):
    def test_popularity_returns_k_and_respects_genre(self):
        recs = PopularityRecommender(DATA).recommend({}, k=5, genre="Horror")
        self.assertEqual(len(recs), 5)
        idx = {m: i for i, m in enumerate(DATA.movies.movieId)}
        for r in recs:
            self.assertIn("Horror", DATA.movies.iloc[idx[r.movie_id]].genres)

    def test_sar_excludes_seen(self):
        recs = SARRecommender(DATA).recommend({MATRIX: 5.0}, k=10)
        self.assertNotIn(MATRIX, [r.movie_id for r in recs])
        self.assertEqual(len(recs), 10)

    def test_content_keywords_without_history(self):
        recs = ContentRecommender(DATA).recommend({}, k=5, keywords="animation children")
        self.assertEqual(len(recs), 5)

    def test_hybrid_cold_start_falls_back_to_popularity(self):
        recs = HYBRID.recommend({}, k=5)
        self.assertEqual(len(recs), 5)
        self.assertIn("popular", recs[0].reason)

    def test_hybrid_explanations_present(self):
        recs = HYBRID.recommend({MATRIX: 5.0, TOY_STORY: 4.5}, k=5)
        self.assertTrue(all(r.reason for r in recs))
        self.assertTrue(any("The Matrix" in r.reason or "Toy Story" in r.reason for r in recs))

    def test_hybrid_no_duplicates(self):
        recs = HYBRID.recommend({MATRIX: 5.0}, k=10)
        ids = [r.movie_id for r in recs]
        self.assertEqual(len(ids), len(set(ids)))


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.bot = CineMate(DATA, HYBRID)

    def test_typo_tolerant_title(self):
        self.assertIn("The Matrix", self.bot.reply("I loved the matrx"))
        self.assertIn(MATRIX, self.bot.liked)

    def test_dislike(self):
        self.bot.reply("I didn't like Titanic")
        self.assertLess(self.bot.liked[TITANIC], 3)

    def test_proactive_after_three_likes(self):
        self.bot.reply("I loved The Matrix")
        self.bot.reply("Toy Story")
        out = self.bot.reply("I liked Inception")
        self.assertIn("you might enjoy", out)

    def test_why_and_more(self):
        self.bot.reply("I loved The Matrix")
        first = self.bot.reply("recommend")
        self.assertIn("1.", first)
        self.assertIn("Genres:", self.bot.reply("why 1"))
        before = set(self.bot.shown)
        self.bot.reply("more")
        self.assertGreater(len(self.bot.shown), len(before))

    def test_rate_numbered_item(self):
        self.bot.reply("recommend")
        mid = self.bot.last_recs[1].movie_id
        self.bot.reply("2 was bad")
        self.assertLess(self.bot.liked[mid], 3)

    def test_forget_and_reset(self):
        self.bot.reply("I loved The Matrix")
        self.bot.reply("forget the matrix")
        self.assertNotIn(MATRIX, self.bot.liked)
        self.bot.reply("Toy Story")
        self.bot.reply("reset")
        self.assertEqual(self.bot.liked, {})

    def test_quiz_flow_ends_in_recommendations(self):
        out = self.bot.reply("quiz me")
        self.assertIn("Did you like", out)
        for _ in range(5):
            out = self.bot.reply("yes")
        self.assertIn("you might enjoy", out)

    def test_unknown_input(self):
        self.assertIn("didn't catch", self.bot.reply("asdf qwerty"))


if __name__ == "__main__":
    unittest.main()
