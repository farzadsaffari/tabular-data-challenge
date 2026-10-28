# WPP Open Intelligence: Data Science Challenge

Click prediction, ad similarity search and a small API on top of both, built on
advertising impression data.

## 1. What I built

The three core objectives of the challenge:

1. **Predict:** a binary classifier that gives the probability that an impression is clicked.
2. **Embed:** a semantic space over the ad creatives, with `find_similar(ad_id, k)`
   that returns the `k` most similar ads.
3. **Serve:** a FastAPI service with two endpoints:
   - `POST /predict`: click probabilities for a list of impressions
   - `GET /similar_ads/{ad_id}?k=5`: the most similar ads to a given ad

Repository structure:

```
wpp-challenge/
├── .github/workflows/run_pipelines.yml   # runs everything on every push
├── data/train.csv                        # SYNTHETIC sample data (see below)
├── scripts/call_api.py                   # calls both API endpoints and prints the answers
├── src/
│   ├── data_preprocessing.py             # loading and cleaning, shared by training and the API
│   ├── model.py                          # feature choices and the sklearn pipeline
│   ├── train.py                          # time-based split, training, evaluation, saving
│   ├── embedding.py                      # ad embeddings and find_similar()
│   └── api.py                            # FastAPI service
├── requirements.txt                      # pinned dependencies (Python 3.11)
└── README.md
```

## 2. Approach and key decisions

### Data cleaning (`src/data_preprocessing.py`)

- **Timestamp:** I profiled the string patterns in the column (digits replaced by
  `9`, then counted). There was a single format, `YYYY-MM-DD HH:MM:SS`, so it is
  parsed explicitly with that format.
- **Categorical text** (`user_segment`, `device_type`, `geo_region`,
  `placement_context`, `advertiser_vertical`) is stripped and lowercased, so
  variants like `Mobile` and `mobile` become one category.
- **`targeting_interests`** (`$`-separated) is split into the columns
  `targeting_interest_1` to `targeting_interest_5`. Interests are sorted within each
  row, so `travel$fitness` and `fitness$travel` give the same result. Empty values
  become missing.
- **Duplicates:** duplicate `impression_id` rows are removed before training.
- The same `clean()` function is used by training and by the API, so both see
  identical inputs.

### Features (`src/model.py`)

| Used | How |
|---|---|
| `user_segment`, `device_type`, `geo_region`, `placement_context`, `advertiser_vertical`, `campaign_id`, `ad_id`, `targeting_interest_1..5` | Ordinal-encoded. Categories unseen in training and missing values get the code `-1`. |
| `advertiser_reported_ctr_pct` | Standard-scaled. See the leakage note below. |

| Not used | Why |
|---|---|
| `impression_id` | An identifier, not a signal. |
| `timestamp` | Used only for the time-based split. |
| `ad_headline`, `ad_description` | Raw text. Ordinal encoding of free text gives arbitrary codes. The text is used for the embeddings instead. |

**Leakage note on `advertiser_reported_ctr_pct`:** it is a vendor-reported CTR, so it
may be computed from the same clicks the model predicts. I kept it, and in my run
during the challenge it did not produce suspiciously high validation scores. That
does not prove there is no leakage: an ad-level average over many impressions moves
very little with one click. The next step is to compare validation scores with and
without it, and to find out how and when the vendor computes it.

### Pipeline and model

- All preprocessing (scaler and encoder) and the model are in **one sklearn
  `Pipeline`**, fitted on the training split only and saved as one file. Serving uses
  exactly the same transformations, with no refitting on new data.
- **Model: `RandomForestClassifier`** (200 trees, `min_samples_leaf=5`,
  `random_state=0`). A sensible, simple baseline for mixed categorical and numeric
  data, with no tuning, as the task asks. Scaling is not needed for a random forest;
  the scaler is kept so the pipeline still works if the model is swapped for one
  that needs it (for example logistic regression).

### Validation and evaluation (`src/train.py`)

- **Time-based split:** the last 7 days are the validation set. The model is used to
  score future traffic, so validation should be later in time than training. A
  random split would let the model learn from the future.
- **Metrics:** clicks are rare, so accuracy is misleading. I report:
  - **ROC-AUC** (random baseline 0.5)
  - **PR-AUC** (baseline: the click rate)
  - **Log loss** (baseline: always predicting the training click rate)
  - **Classification report** at the default 0.5 threshold, which shows the class
    imbalance (support) and how many clicks are caught at that threshold
- Random forest probabilities are not calibrated, so a score of 0.2 is not exactly a
  20% click chance. This matters for log loss and bidding, not for the ranking
  metrics.

### Embeddings (`src/embedding.py`)

- **One embedding per ad:** impressions repeat the same ad, so the ads are
  deduplicated by `ad_id` first. Otherwise the nearest neighbours would be copies of
  the same ad.
- **Text:** `ad_headline` and `ad_description` combined, since both describe the
  creative.
- **Model:** `all-MiniLM-L6-v2` (sentence-transformers). Small, fast, and a common
  baseline for semantic similarity.
- **Similarity:** embeddings are normalised, so the dot product equals cosine
  similarity. The ad itself is excluded from its own results. An unknown `ad_id`
  raises an error (404 in the API).
- The embeddings are saved to `models/`, so the API loads them instead of
  recomputing them at startup.

### API (`src/api.py`)

- Loads the trained pipeline and the embeddings once at startup.
- `POST /predict` takes `{"records": [...]}`, where each record has the raw fields
  `clean()` needs: `impression_id`, `timestamp`, `user_segment`, `device_type`,
  `geo_region`, `placement_context`, `advertiser_vertical`, `campaign_id`, `ad_id`,
  `targeting_interests`, `advertiser_reported_ctr_pct`. The label `click` is not
  sent, since it is unknown at prediction time.
- Input validation is minimal (records are plain dictionaries), in line with the
  task's scope.

### Environment

`requirements.txt` pins every main package for Python 3.11, so runs are reproducible.
In GitHub Actions, the CPU-only build of `torch` is installed first, to avoid
downloading GPU libraries.

## 3. How to run it

### Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Data

- **Real data:** put the challenge's `data.csv` in `data/`, so the path is
  `data/data.csv`. It is not included in the repo.
- **Sample data:** `data/train.csv` is a **synthetic** sample with the same 14
  columns as `data.csv`, generated so the pipeline can be run end to end without the
  real data. Category values, ad texts and click rates are invented. **It is not the
  real data, and the metrics on it mean nothing.**

### Run everything

Run from the repository root. Replace `data/data.csv` with `data/train.csv` to use
the synthetic sample.

```bash
python -m src.train --data data/data.csv        # trains, prints metrics, saves models/click_pipeline.joblib
python -m src.embedding --data data/data.csv    # builds and saves the ad embeddings
uvicorn src.api:app                             # starts the API, leave it running
```

In a second terminal:

```bash
python scripts/call_api.py --data data/data.csv
```

This sends three rows to `/predict` and asks `/similar_ads` for the neighbours of
the first ad. You can also open http://127.0.0.1:8000/docs to try both endpoints in
the browser.

The first embedding run downloads the sentence-transformers model (about 90 MB), so
it needs internet access.

### GitHub Actions

`.github/workflows/run_pipelines.yml` runs the same steps on `data/train.csv` on
every push: install, train, build embeddings, start the API, and call both
endpoints. A failed step fails the run.

## 4. Known limitations and next steps

- **Not run on the real data after restructuring.** The challenge was written as a
  single script on Coderbyte. This repo is the restructured version, and it has been
  run on the synthetic sample only.
- **Leakage check** on `advertiser_reported_ctr_pct`, as described above.
- **Interests:** positional columns are simple, but the same interest in different
  positions is seen as different features. Multi-hot encoding is the natural
  improvement. `MAX_INTERESTS` is fixed at 5, and extra interests are dropped.
- **Text features for the classifier:** the ad embeddings could be used as model
  features.
- **Calibration** of the predicted probabilities.
- **API hardening:** a typed request schema, tests and a Dockerfile.
- **Scale:** brute-force similarity is fine for a few thousand ads. For millions,
  a vector index (for example FAISS or Qdrant) would replace it.

## Background

The challenge was started on Coderbyte, where I could only work in a single script,
and the platform stopped running code partway through. I did not finish the API
during the challenge. As requested by the team, this repository is the structured
version of my work, with the API completed and instructions for running it.