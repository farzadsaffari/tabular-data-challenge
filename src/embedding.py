"""Ad text embeddings and find_similar().
Run from repo root: python -m src.embed --data data/data.csv"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from src.data_preprocessing import load_raw

MODEL_NAME = "all-MiniLM-L6-v2"
ART_DIR = Path("models")


class SimilarAds:
    def __init__(self, ad_ids, texts, emb):
        self.ad_ids, self.texts, self.emb = ad_ids, texts, emb
        self.pos = {a: i for i, a in enumerate(ad_ids)}

    def find_similar(self, ad_id: str, k: int = 5) -> list[dict]:
        if ad_id not in self.pos:
            raise KeyError(ad_id)
        i = self.pos[ad_id]
        scores = self.emb @ self.emb[i]     # cosine similarity, rows are normalised
        scores[i] = -np.inf                 # exclude the ad itself
        top = np.argsort(-scores)[:k]
        return [{"ad_id": self.ad_ids[j], "text": self.texts[j],
                 "similarity": float(scores[j])} for j in top]


def build_ad_table(df: pl.DataFrame) -> pl.DataFrame:
    # Impressions repeat each ad, so embed each ad_id once.
    return (
        df.select(["ad_id", "ad_headline", "ad_description"])
          .with_columns(pl.col("ad_id").cast(pl.String))
          .unique(subset="ad_id", keep="first")
          .with_columns(text=(pl.col("ad_headline").fill_null("") + ". "
                              + pl.col("ad_description").fill_null("")).str.strip_chars())
    )


def build_and_save(data_path: str, out_dir: Path = ART_DIR) -> None:
    from sentence_transformers import SentenceTransformer  # heavy import, only at build time

    ads = build_ad_table(load_raw(data_path))
    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(ads["text"].to_list(), normalize_embeddings=True, show_progress_bar=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ad_embeddings.npy", np.asarray(emb, dtype=np.float32))
    (out_dir / "ad_index.json").write_text(json.dumps(
        {"ad_ids": ads["ad_id"].to_list(), "texts": ads["text"].to_list()}))
    print(f"embedded {ads.height} ads")


def load_index(out_dir: Path = ART_DIR) -> SimilarAds:
    emb = np.load(out_dir / "ad_embeddings.npy")
    meta = json.loads((out_dir / "ad_index.json").read_text())
    return SimilarAds(meta["ad_ids"], meta["texts"], emb)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.csv")
    build_and_save(ap.parse_args().data)