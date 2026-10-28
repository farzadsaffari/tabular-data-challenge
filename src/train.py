"""Time-based split, train, evaluate, save.
Run from repo root: python -m src.train --data data/data.csv"""
import argparse
from datetime import timedelta
from pathlib import Path

import joblib
import polars as pl
from sklearn.metrics import (average_precision_score, classification_report,
                             log_loss, roc_auc_score)

from src.data_preprocessing import CATEGORICAL, NUMERIC, clean, load_raw
from src.model import build_pipeline

TARGET = "click"
VAL_DAYS = 7


def time_split(df: pl.DataFrame, days: int = VAL_DAYS):
    cutoff = df["timestamp"].max() - timedelta(days=days)
    return df.filter(pl.col("timestamp") < cutoff), df.filter(pl.col("timestamp") >= cutoff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.csv")
    ap.add_argument("--out", default="models/click_pipeline.joblib")
    args = ap.parse_args()

    df = clean(load_raw(args.data)).unique(subset="impression_id", keep="first").sort("timestamp")
    assert df["timestamp"].null_count() == 0, "some timestamps did not parse"

    train, val = time_split(df)
    cols = NUMERIC + CATEGORICAL
    X_tr, y_tr = train.select(cols).to_pandas(), train[TARGET].to_numpy()
    X_va, y_va = val.select(cols).to_pandas(), val[TARGET].to_numpy()
    print(f"train {len(y_tr)} rows, click rate {y_tr.mean():.4f}")
    print(f"val   {len(y_va)} rows, click rate {y_va.mean():.4f}")

    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_va)[:, 1]

    print(f"ROC-AUC  model {roc_auc_score(y_va, p):.4f}  (baseline 0.5)")
    print(f"PR-AUC   model {average_precision_score(y_va, p):.4f}  (baseline {y_va.mean():.4f})")
    print(f"log loss model {log_loss(y_va, p):.4f}  "
          f"(baseline {log_loss(y_va, [y_tr.mean()] * len(y_va)):.4f})")

    y_pred = pipe.predict(X_va)   # default 0.5 threshold
    print(classification_report(y_va, y_pred, digits=3, zero_division=0))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()