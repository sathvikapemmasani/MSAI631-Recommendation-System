# CineMate – Conversational Movie Recommender

MSAI-631-M80 · Artificial Intelligence for Human-Computer Interaction · University of the Cumberlands  
Author: Sathvika Pemmasani

CineMate is a hybrid movie recommender you talk to in plain English. It blends
item-to-item collaborative filtering with content-based filtering over genres and
community tags, explains every suggestion, and proactively offers picks once it
knows enough about you.

## Foundation and credit
* **Starting sample:** the SAR (Simple Algorithm for Recommendation) MovieLens
  quick-start from Microsoft's open-source *Recommenders* project
  (https://github.com/recommenders-team/recommenders). I re-implemented the core
  idea (Jaccard item co-occurrence scored against a user's history) in plain
  NumPy/SciPy so it would run anywhere.
* **Data:** MovieLens latest-small (Harper & Konstan, 2015), GroupLens Research.
  The data is not redistributed here; download it with the script below.

## What I changed / added
| Area | Original sample | CineMate |
|---|---|---|
| Users | Existing dataset users only | Anyone: works from ratings given in the chat |
| Model | Collaborative only | Hybrid: SAR + TF-IDF content (genres + tags) + popularity prior |
| Cold start | Not handled | Popularity fallback + "quiz me" onboarding |
| Ranking | Pure score | Diversity re-ranking (Maximal Marginal Relevance) |
| Interface | Notebook cells | Conversational agent with typo-tolerant titles |
| Transparency | None | One-line reason per pick, "why N" for details |
| Proactivity | None | Offers recommendations after 3 likes without being asked |

## Quick start
```bash
pip install -r requirements.txt
python tools/download_data.py      # fetch MovieLens (~1 MB)
python app.py                      # chat with CineMate
python -m cinemate.evaluate        # offline evaluation
python -m unittest discover -s tests -v
```
In Google Colab, open `CineMate_Colab.ipynb` and run the cells top to bottom.

## Things to say
`I loved The Matrix` · `I didn't like Titanic` · `recommend` · `something funny` ·
`movies like Up` · `why 2` · `more` · `2 was bad` · `my list` · `forget Titanic` ·
`quiz me` · `reset` · `bye`

## Results (200 test users, 80/20 per-user split)
See `docs/evaluation_results.txt`. The hybrid matches the SAR baseline on
Precision@10 (0.144 vs 0.143) while producing more diverse lists, and it can
answer free-text and cold-start requests that SAR alone cannot.

## Project layout
```
cinemate/data_loader.py   load + clean MovieLens
cinemate/models.py        Popularity, SAR, Content, Hybrid recommenders
cinemate/agent.py         CineMate conversational agent
cinemate/evaluate.py      offline evaluation
app.py                    console chat
tests/test_cinemate.py    17 unit tests
tools/download_data.py    dataset download
```

## AI use disclosure
Claude (Anthropic) was used to help write and debug code and to draft documentation.
All code was run, tested, and reviewed by the author.
