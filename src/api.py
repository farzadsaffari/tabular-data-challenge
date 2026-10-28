"""FastAPI service. Run from repo root: uvicorn src.api:app --reload
Status: written but NOT tested end to end. See README."""
import joblib
import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_preprocessing import CATEGORICAL, NUMERIC, clean
from src.embedding import load_index

app = FastAPI(title="Ad model")
state: dict = {}


@app.on_event("startup")
def load_artifacts():
    state["pipe"] = joblib.load("models/click_pipeline.joblib")
    state["similar"] = load_index()


class PredictRequest(BaseModel):
    # shaped like the CSV rows, with timestamp as "YYYY-MM-DD HH:MM:SS".
    records: list[dict]


@app.post("/predict")
def predict(req: PredictRequest):
    raw = pl.DataFrame(req.records, infer_schema_length=None)
    raw = raw.with_columns(pl.col("campaign_id", "ad_id").cast(pl.String))
    df = clean(raw)
    X = df.select(NUMERIC + CATEGORICAL).to_pandas()
    probs = state["pipe"].predict_proba(X)[:, 1]
    return {"predictions": [
        {"impression_id": i, "p_click": float(p)}
        for i, p in zip(raw["impression_id"].to_list(), probs)
    ]}


@app.get("/similar_ads/{ad_id}")
def similar_ads(ad_id: str, k: int = 5):
    try:
        return {"ad_id": ad_id, "results": state["similar"].find_similar(ad_id, k)}
    except KeyError:
        raise HTTPException(status_code=404, detail=f"unknown ad_id {ad_id}")
