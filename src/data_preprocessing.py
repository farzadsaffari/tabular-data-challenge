"""Loading and cleaning. clean() is shared by training and the API,
so both see identical inputs."""
import polars as pl

# Number of interest columns. It is set it from the TRAINING data, and  it will be kept fixed so
# training and the API always produce the same columns.  it can be checked with:
#   train.select(pl.col("targeting_interests").str.split("$").list.len().max())
# Rows with more target interests than this keep only the first MAX_INTERESTS (alphabetically).
MAX_INTERESTS = 5

INTEREST_COLS = [f"targeting_interest_{i + 1}" for i in range(MAX_INTERESTS)]

CATEGORICAL = ["user_segment", "device_type", "geo_region", "placement_context",
               "advertiser_vertical", "campaign_id", "ad_id"] + INTEREST_COLS
NUMERIC = ["advertiser_reported_ctr_pct"]
LOWERCASE_COLS = ["user_segment", "device_type", "geo_region",
                  "placement_context", "advertiser_vertical"]


def load_raw(path: str) -> pl.DataFrame:
    return pl.read_csv(path, schema_overrides={
        "timestamp": pl.String,
        "targeting_interests": pl.String,
    })


def split_interests(df: pl.DataFrame) -> pl.DataFrame:
    """'travel$fitness$investing' -> targeting_interest_1..N.
    Interests are lowercased and sorted within each row, so the order they were
    written in does not matter. Empty strings become nulls."""
    cleaned = (
        pl.when(pl.col("targeting_interests").str.strip_chars() == "")
          .then(None)
          .otherwise(pl.col("targeting_interests").str.strip_chars().str.to_lowercase())
    )
    items = cleaned.str.split("$").list.sort()
    return df.with_columns(
        [items.list.get(i, null_on_oob=True).alias(f"targeting_interest_{i + 1}")
         for i in range(MAX_INTERESTS)]
    ).drop("targeting_interests")


def clean(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("timestamp").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False),
        *[pl.col(c).str.strip_chars().str.to_lowercase() for c in LOWERCASE_COLS],
    )
    df = split_interests(df)
    # Make every categorical column a string. Ids can be read as integers, and
    # mixing ints and strings breaks the encoder.
    return df.with_columns([pl.col(c).cast(pl.String) for c in CATEGORICAL])