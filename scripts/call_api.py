"""Call the running API with a few rows from a CSV, and print the answers.

The API must already be running (uvicorn src.api:app).
Run from repo root:
    python scripts/call_api.py --data data/sample_data.csv --url http://127.0.0.1:8000
Uses only the standard library, so it needs no extra packages.
"""
import argparse
import csv
import json
import urllib.request

# The raw fields clean() needs. `click` is not sent: the label is not known at
# prediction time. The ad text is not sent: the classifier does not use it.
REQUEST_FIELDS = ["impression_id", "timestamp", "user_segment", "device_type",
                  "geo_region", "placement_context", "advertiser_vertical",
                  "campaign_id", "ad_id", "targeting_interests",
                  "advertiser_reported_ctr_pct"]


def read_records(path: str, n: int) -> list[dict]:
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if any(row[c] == "" for c in REQUEST_FIELDS):
                continue  # keep the demo simple: complete rows only
            record = {c: row[c] for c in REQUEST_FIELDS}
            # CSV values are text, so turn the numeric field into a number.
            record["advertiser_reported_ctr_pct"] = float(record["advertiser_reported_ctr_pct"])
            records.append(record)
            if len(records) == n:
                break
    return records


def call(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:  # raises on 4xx/5xx
        return json.loads(response.read())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sample_data.csv")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    records = read_records(args.data, args.n)

    print("POST /predict")
    print(json.dumps(call(f"{args.url}/predict", {"records": records}), indent=2))

    ad_id = records[0]["ad_id"]
    print(f"\nGET /similar_ads/{ad_id}?k=3")
    print(json.dumps(call(f"{args.url}/similar_ads/{ad_id}?k=3"), indent=2))


if __name__ == "__main__":
    main()