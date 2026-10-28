"""Feature choices and the sklearn pipeline."""
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.data_preprocessing import CATEGORICAL, NUMERIC

# Deliberately NOT used as features:
# - impression_id: an identifier, not a signal.
# - timestamp: only used for the time-based split.
# - ad_headline, ad_description: raw text, not used by the classifier yet. The ordinal
#   encoder only works if there are few unique strings. Text is used in embedding.py.
#
# Used, with a caveat:
# - advertiser_reported_ctr_pct: vendor-reported, so it may be derived from the same
#   clicks. Included, and no sign of inflated validation metrics, but that cannot
#   prove there is no leakage. See README.
#
# Used:
# - targeting_interests: split into targeting_interest_1..N in data_preprocessing.py.


def build_pipeline() -> Pipeline:
    # Categorical columns -> integer codes. Fitted on train only, inside the pipeline.
    # Categories unseen in training and missing values both get the code -1.
    cat_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),   # fit on train only, inside the pipeline
        ("cat", cat_encoder, CATEGORICAL),
    ])
    return Pipeline([
        ("pre", pre),
        # Missing numeric values (NaN) are handled by the forest itself.
        # This needs scikit-learn >= 1.4.
        ("model", RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=0,
        )),
    ])