from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List
import hashlib
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creators.csv"

VALID_PLATFORMS = {"youtube", "instagram", "tiktok"}
VALID_COUNTRIES = {"All", "KR", "Global"}
KOREA_LOCATION_KEYS = {"kr", "south korea", "korea", "republic of korea", "korea, republic of"}

# The uploaded CSV contains these real columns. The file does NOT contain a
# historical campaign outcome, so the demo builds a transparent proxy target.
FEATURES = ["brand_fit", "engagement_rate", "followers_m", "verified_flag"]

FEATURE_LABELS = {
    "brand_fit": "Brand Fit Score",
    "engagement_rate": "Engagement Rate",
    "followers_m": "Followers",
    "verified_flag": "Verified Creator",
}

DEFAULT_BRAND_WEIGHT = 0.50
FIT_SCALE_LABEL = "Ordinal 5-level heuristic scale (1.00 / 0.75 / 0.50 / 0.25 / 0.00)"

BRAND_PROFILES: Dict[str, Dict[str, float]] = {
    "BVLGARI": {
        "fashion": 1.00,
        "fashion_&_style": 1.00,
        "beauty": 0.75,
        "lifestyle": 0.50,
        "travel": 0.25,
        "music": 0.25,
        "fitness": 0.00,
        "food": 0.00,
        "business": 0.00,
        "tech": 0.00,
        "gaming": 0.00,
        "education": 0.00,
        "finance": 0.00,
        "comedy": 0.00,
    },
    "Gucci": {
        "fashion": 1.00,
        "fashion_&_style": 1.00,
        "beauty": 0.75,
        "lifestyle": 0.75,
        "music": 0.50,
        "travel": 0.50,
        "comedy": 0.25,
        "fitness": 0.25,
        "food": 0.25,
        "business": 0.00,
        "tech": 0.00,
        "gaming": 0.00,
        "education": 0.00,
        "finance": 0.00,
    },
    "Dior": {
        "fashion": 1.00,
        "fashion_&_style": 1.00,
        "beauty": 1.00,
        "lifestyle": 0.75,
        "travel": 0.50,
        "music": 0.25,
        "fitness": 0.25,
        "food": 0.00,
        "business": 0.00,
        "tech": 0.00,
        "gaming": 0.00,
        "education": 0.00,
        "finance": 0.00,
        "comedy": 0.00,
    },
    "Nike": {
        "fitness": 1.00,
        "lifestyle": 0.75,
        "fashion": 0.50,
        "fashion_&_style": 0.50,
        "travel": 0.50,
        "music": 0.25,
        "gaming": 0.25,
        "comedy": 0.25,
        "food": 0.25,
        "business": 0.00,
        "tech": 0.25,
        "education": 0.00,
        "finance": 0.00,
        "beauty": 0.25,
    },
}

CAMPAIGN_PROFILES: Dict[str, Dict[str, float]] = {
    "Luxury / Fashion": {
        "fashion": 1.00,
        "fashion_&_style": 1.00,
        "beauty": 0.75,
        "lifestyle": 0.50,
        "travel": 0.25,
        "music": 0.25,
        "fitness": 0.00,
        "food": 0.00,
        "business": 0.00,
        "tech": 0.00,
        "gaming": 0.00,
        "education": 0.00,
        "finance": 0.00,
        "comedy": 0.00,
    },
    "Jewelry": {
        "fashion": 1.00,
        "fashion_&_style": 1.00,
        "beauty": 0.75,
        "lifestyle": 0.75,
        "travel": 0.25,
        "music": 0.25,
        "fitness": 0.00,
        "food": 0.00,
        "business": 0.00,
        "tech": 0.00,
        "gaming": 0.00,
        "education": 0.00,
        "finance": 0.25,
        "comedy": 0.00,
    },
    "Beauty": {
        "beauty": 1.00,
        "fashion": 0.75,
        "fashion_&_style": 0.75,
        "lifestyle": 0.75,
        "travel": 0.25,
        "fitness": 0.25,
        "music": 0.25,
        "food": 0.00,
        "business": 0.00,
        "tech": 0.00,
        "gaming": 0.00,
        "education": 0.00,
        "finance": 0.00,
        "comedy": 0.00,
    },
    "Lifestyle": {
        "lifestyle": 1.00,
        "travel": 0.75,
        "fashion": 0.50,
        "fashion_&_style": 0.50,
        "beauty": 0.50,
        "fitness": 0.50,
        "food": 0.50,
        "music": 0.25,
        "comedy": 0.25,
        "business": 0.25,
        "tech": 0.25,
        "gaming": 0.25,
        "education": 0.25,
        "finance": 0.00,
    },
}



@dataclass
class DataBundle:
    creators: pd.DataFrame
    repaired_rows: int
    raw_encoding: str


@dataclass
class ContextModel:
    brand: str
    campaign: str
    country: str
    scored: pd.DataFrame
    logistic: Pipeline
    tree: DecisionTreeClassifier
    proxy_positive_rate: float
    training_size: int


def _read_uploaded_csv(path: Path) -> pd.DataFrame:
    """Read the merged 5,300-row creator CSV safely.

    The Korea-augmented file is UTF-8 with BOM, while older source copies may
    be legacy encoded. Try UTF-8 first so Korean names remain readable, then
    fall back to latin-1 for byte-preserving compatibility.
    """
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def _repair_shifted_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Repair rows where the platform token was accidentally appended to name.

    In 42 rows of the uploaded file, e.g. `Shraddha ??instagram,lifestyle,...`,
    the comma between name and platform is missing. That shifts all following
    values one column to the left. This function detects and repairs that pattern.
    """
    out = df.copy()
    # Use object dtype during repair because malformed rows temporarily place
    # strings into columns that pandas inferred as numeric.
    for col in ["platform", "niche", "followers", "avg_engagement_rate", "location", "verified", "fake_followers_pct"]:
        out[col] = out[col].astype(object)
    invalid = ~out["platform"].astype(str).str.lower().isin(VALID_PLATFORMS)
    repaired = 0

    for idx in out.index[invalid]:
        name = str(out.at[idx, "name"])
        match = re.search(r"(youtube|instagram|tiktok)\s*$", name, flags=re.I)
        if not match:
            continue

        # Save shifted values before overwriting them.
        old_platform = out.at[idx, "platform"]
        old_niche = out.at[idx, "niche"]
        old_followers = out.at[idx, "followers"]
        old_engagement = out.at[idx, "avg_engagement_rate"]
        old_location = out.at[idx, "location"]
        old_verified = out.at[idx, "verified"]

        out.at[idx, "name"] = name[: match.start()].rstrip(" ?|-/")
        out.at[idx, "platform"] = match.group(1).lower()
        out.at[idx, "niche"] = old_platform
        out.at[idx, "followers"] = old_niche
        out.at[idx, "avg_engagement_rate"] = old_followers
        out.at[idx, "location"] = old_engagement
        out.at[idx, "verified"] = old_location
        out.at[idx, "fake_followers_pct"] = old_verified
        repaired += 1

    return out, repaired


def load_data() -> DataBundle:
    raw = _read_uploaded_csv(DATA_PATH)
    df, repaired = _repair_shifted_rows(raw)

    # Standardize fields used by the dashboard/model.
    df["handle"] = df["handle"].astype(str).str.strip()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["platform"] = df["platform"].astype(str).str.strip().str.lower()
    df["niche"] = (
        df["niche"].fillna("unknown").astype(str).str.strip().str.lower()
    )
    df["followers"] = pd.to_numeric(df["followers"], errors="coerce")
    df["avg_engagement_rate"] = pd.to_numeric(
        df["avg_engagement_rate"], errors="coerce"
    )
    df["fake_followers_pct"] = pd.to_numeric(
        df["fake_followers_pct"], errors="coerce"
    )

    # Keep unknown verification distinct from a known FALSE value. Missing
    # verification is imputed by platform median so Korea rows are not
    # automatically penalized merely because the public source omitted it.
    verified_text = df["verified"].fillna("").astype(str).str.upper().str.strip()
    true_values = {"TRUE", "1", "YES", "Y"}
    false_values = {"FALSE", "0", "NO", "N"}
    verified_raw = pd.Series(np.nan, index=df.index, dtype=float)
    verified_raw[verified_text.isin(true_values)] = 1.0
    verified_raw[verified_text.isin(false_values)] = 0.0
    df["verified_known"] = verified_raw.notna()

    # Drop only records that cannot participate at all. Engagement and verified
    # may be missing in public rankings, so those fields are imputed below.
    df = df[df["platform"].isin(VALID_PLATFORMS)].copy()
    df = df.dropna(subset=["followers"]).copy()
    df["followers"] = df["followers"].clip(lower=0)
    df["avg_engagement_rate"] = df["avg_engagement_rate"].clip(lower=0, upper=50)

    # Country grouping requested by the dashboard. The 300 added Korea records
    # use 'South Korea'; existing rows using 'KR' are also treated as Korea.
    location_key = df["location"].fillna("").astype(str).str.strip().str.lower()
    df["country_group"] = np.where(location_key.isin(KOREA_LOCATION_KEYS), "KR", "Global")

    # Platform-median imputation preserves Korean YouTube rows whose source did
    # not publish a comparable engagement-rate metric. The UI marks imputed
    # values with an asterisk.
    df["engagement_imputed"] = df["avg_engagement_rate"].isna()
    platform_engagement_median = df.groupby("platform")["avg_engagement_rate"].transform("median")
    overall_engagement_median = float(df["avg_engagement_rate"].median())
    df["engagement_rate"] = (
        df["avg_engagement_rate"]
        .fillna(platform_engagement_median)
        .fillna(overall_engagement_median)
        .astype(float)
    )

    # Verification uses the platform median of known records for missing rows.
    known_verified = verified_raw.reindex(df.index)
    tmp = pd.DataFrame({"platform": df["platform"], "value": known_verified})
    medians = tmp.groupby("platform")["value"].transform("median")
    overall_verified = float(known_verified.median()) if known_verified.notna().any() else 0.0
    df["verified_imputed"] = known_verified.isna()
    df["verified_flag"] = known_verified.fillna(medians).fillna(overall_verified).round().astype(int)

    df["followers_m"] = df["followers"].astype(float) / 1_000_000.0

    return DataBundle(
        creators=df.reset_index(drop=True),
        repaired_rows=repaired,
        raw_encoding="utf-8-sig (legacy fallback supported)",
    )


DATA = load_data()


def _profile_score(niche: str, profile: Dict[str, float]) -> float:
    key = str(niche).lower().strip()
    if key in profile:
        return float(profile[key])
    if "fashion" in key:
        return float(profile.get("fashion", 0.75))
    return 0.00


def _normalize_brand_weight(brand_weight: float) -> float:
    return float(np.clip(float(brand_weight), 0.0, 1.0))


def compute_brand_fit(
    niche: str,
    brand: str,
    campaign: str,
    brand_weight: float = DEFAULT_BRAND_WEIGHT,
) -> float:
    brand_weight = _normalize_brand_weight(brand_weight)
    campaign_weight = 1.0 - brand_weight
    brand_score = _profile_score(niche, BRAND_PROFILES[brand])
    campaign_score = _profile_score(niche, CAMPAIGN_PROFILES[campaign])
    return float(np.clip(brand_weight * brand_score + campaign_weight * campaign_score, 0.0, 1.0))


def enrich_context(
    df: pd.DataFrame,
    brand: str,
    campaign: str,
    brand_weight: float = DEFAULT_BRAND_WEIGHT,
) -> pd.DataFrame:
    out = df.copy()
    out["brand_fit"] = out["niche"].map(
        lambda n: compute_brand_fit(n, brand, campaign, brand_weight)
    )
    return out


def _stable_noise(handle: str, brand: str, campaign: str) -> float:
    """Deterministic pseudo-random noise in roughly [-1, 1]."""
    token = f"{handle}|{brand}|{campaign}".encode("utf-8", errors="ignore")
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64
    return (value - 0.5) * 2.0


def _proxy_target(df: pd.DataFrame, brand: str, campaign: str) -> pd.Series:
    """Create a reproducible proxy label because the CSV has no true outcome.

    This is intentionally transparent and should be replaced with a historical
    campaign outcome (selected, converted, ROAS success, etc.) when available.
    """
    reach = np.log1p(df["followers_m"].clip(lower=0))
    engagement = np.minimum(df["engagement_rate"], 20.0) / 20.0
    noise = np.array([
        _stable_noise(h, brand, campaign) for h in df["handle"].astype(str)
    ])

    latent = (
        -4.00
        + 7.00 * df["brand_fit"].to_numpy()
        + 1.00 * engagement.to_numpy()
        + 0.10 * reach.to_numpy()
        + 0.20 * df["verified_flag"].to_numpy()
        + 0.45 * noise
    )
    p = 1.0 / (1.0 + np.exp(-latent))

    # Deterministic Bernoulli-like label using a second stable hash stream.
    uniforms = []
    for h in df["handle"].astype(str):
        token = f"target|{h}|{brand}|{campaign}".encode("utf-8", errors="ignore")
        uniforms.append(int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2**64)
    return pd.Series(np.array(uniforms) < p, index=df.index, dtype=int)


def filter_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    if country not in VALID_COUNTRIES:
        raise ValueError(f"Unknown country filter: {country}")
    if country == "All":
        return df.copy()
    return df[df["country_group"] == country].copy()


@lru_cache(maxsize=512)
def train_context(
    brand: str,
    campaign: str,
    country: str = "All",
    brand_weight: float = DEFAULT_BRAND_WEIGHT,
    min_followers: int = 0,
    min_engagement: float = 0.0,
    tree_depth: int = 4,
) -> ContextModel:
    if brand not in BRAND_PROFILES:
        raise ValueError(f"Unknown brand: {brand}")
    if campaign not in CAMPAIGN_PROFILES:
        raise ValueError(f"Unknown campaign: {campaign}")
    if country not in VALID_COUNTRIES:
        raise ValueError(f"Unknown country filter: {country}")

    country_df = filter_country(DATA.creators, country)

    min_followers = max(0, int(min_followers))
    min_engagement = max(0.0, float(min_engagement))
    tree_depth = int(np.clip(int(tree_depth), 2, 6))

    # Business scenario filters are applied before model fitting so the
    # Logistic Regression and Decision Tree describe the same eligible pool.
    country_df = country_df[
        (country_df["followers"] >= min_followers)
        & (country_df["engagement_rate"] >= min_engagement)
    ].copy()

    if len(country_df) < 20:
        raise ValueError(
            "Not enough creators after scenario filters. "
            "Lower Minimum Followers or Minimum Engagement."
        )

    brand_weight = round(_normalize_brand_weight(brand_weight), 2)
    df = enrich_context(country_df, brand, campaign, brand_weight)
    y = _proxy_target(df, brand, campaign)
    X = df[FEATURES]

    logistic = Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])
    logistic.fit(X, y)

    # Scale leaf size to the selected country cohort. This keeps the Global
    # tree stable while allowing a meaningful Korea-only tree (~300 records).
    min_leaf = max(8, min(120, int(round(len(df) * 0.08))))
    tree = DecisionTreeClassifier(
        max_depth=tree_depth,
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(X, y)

    scored = df.copy()
    scored["selection_probability"] = logistic.predict_proba(X)[:, 1]
    scored["proxy_target"] = y

    return ContextModel(
        brand=brand,
        campaign=campaign,
        country=country,
        scored=scored,
        logistic=logistic,
        tree=tree,
        proxy_positive_rate=float(y.mean()),
        training_size=int(len(df)),
    )


def score_creators(
    brand: str,
    campaign: str,
    platform: str = "All",
    country: str = "All",
    brand_weight: float = DEFAULT_BRAND_WEIGHT,
    min_followers: int = 0,
    min_engagement: float = 0.0,
    tree_depth: int = 4,
) -> tuple[ContextModel, pd.DataFrame]:
    model = train_context(
        brand,
        campaign,
        country,
        round(_normalize_brand_weight(brand_weight), 2),
        int(min_followers),
        round(float(min_engagement), 2),
        int(tree_depth),
    )
    scored = model.scored.copy()

    if platform and platform != "All":
        platform_key = platform.strip().lower()
        scored = scored[scored["platform"] == platform_key].copy()

    scored = scored.sort_values(
        ["selection_probability", "brand_fit", "engagement_rate", "followers"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    return model, scored


def _compact_count(value: float) -> str:
    n = float(value)
    if n >= 1_000_000:
        text = f"{n / 1_000_000:.2f}".rstrip("0").rstrip(".")
        return f"{text}M"
    if n >= 1_000:
        text = f"{n / 1_000:.1f}".rstrip("0").rstrip(".")
        return f"{text}K"
    return f"{n:.0f}"


def _format_threshold(feature: str, threshold: float) -> str:
    if feature == "engagement_rate":
        return f"{threshold:.2f}%"
    if feature == "followers_m":
        return _compact_count(threshold * 1_000_000)
    if feature == "brand_fit":
        return f"{threshold:.2f}"
    if feature == "verified_flag":
        return f"{threshold:.2f}"
    return f"{threshold:.2f}"


def _format_actual(feature: str, value: float) -> str:
    if feature == "verified_flag":
        return "Yes" if value >= 0.5 else "No"
    return _format_threshold(feature, value)


def export_tree(model: DecisionTreeClassifier, decision_threshold: float = 0.50) -> dict:
    tree_ = model.tree_

    def node_dict(node_id: int, depth: int = 0) -> dict:
        left = int(tree_.children_left[node_id])
        right = int(tree_.children_right[node_id])
        raw_counts = tree_.value[node_id][0]
        total = float(np.sum(raw_counts)) or 1.0
        positive_prob = float(raw_counts[1] / total) if len(raw_counts) > 1 else 0.0

        item = {
            "id": int(node_id),
            "depth": depth,
            "samples": int(tree_.n_node_samples[node_id]),
            "positive_probability": positive_prob,
            "is_leaf": left == right,
        }

        if left == right:
            item["label"] = "Suitable" if positive_prob >= decision_threshold else "Not Suitable"
            return item

        feature = FEATURES[int(tree_.feature[node_id])]
        threshold = float(tree_.threshold[node_id])
        item.update({
            "feature": feature,
            "feature_label": FEATURE_LABELS[feature],
            "threshold": threshold,
            "threshold_label": _format_threshold(feature, threshold),
            "left": node_dict(left, depth + 1),
            "right": node_dict(right, depth + 1),
        })
        return item

    return node_dict(0)


def creator_record(row: pd.Series) -> dict:
    return {
        "handle": str(row["handle"]),
        "name": str(row.get("name", "")),
        "platform": str(row["platform"]).title(),
        "niche": str(row["niche"]).replace("_&_", " & ").replace("_", " ").title(),
        "followers": int(round(float(row["followers"]))),
        "engagement_rate": float(row["engagement_rate"]),
        "engagement_imputed": bool(row.get("engagement_imputed", False)),
        "verified": bool(int(row["verified_flag"])),
        "verified_imputed": bool(row.get("verified_imputed", False)),
        "country": str(row.get("country_group", "Global")),
        "location": str(row.get("location", "")) if pd.notna(row.get("location", "")) else "",
        "brand_fit": float(row["brand_fit"]),
        "selection_probability": float(row["selection_probability"]),
        "rank": int(row["rank"]),
    }


def top_records(scored: pd.DataFrame, n: int = 10) -> List[dict]:
    return [creator_record(row) for _, row in scored.head(n).iterrows()]


def logistic_explanation(model: ContextModel, row: pd.DataFrame) -> dict:
    """Expose the exact predict_proba calculation for one selected creator.

    The fitted Pipeline is: StandardScaler -> LogisticRegression.  Therefore
    z = intercept + sum(coef_j * standardized_x_j), and p = sigmoid(z).
    """
    scaler: StandardScaler = model.logistic.named_steps["scale"]
    classifier: LogisticRegression = model.logistic.named_steps["model"]

    raw = row[FEATURES].iloc[0].astype(float).to_numpy()
    means = scaler.mean_.astype(float)
    scales = scaler.scale_.astype(float)
    standardized = (raw - means) / scales
    coefficients = classifier.coef_[0].astype(float)
    intercept = float(classifier.intercept_[0])
    contributions = coefficients * standardized
    logit = float(intercept + contributions.sum())
    probability = float(1.0 / (1.0 + np.exp(-logit)))

    features = []
    for i, feature in enumerate(FEATURES):
        features.append({
            "feature": feature,
            "feature_label": FEATURE_LABELS[feature],
            "raw_value": float(raw[i]),
            "raw_label": _format_actual(feature, float(raw[i])),
            "mean": float(means[i]),
            "scale": float(scales[i]),
            "standardized": float(standardized[i]),
            "coefficient": float(coefficients[i]),
            "contribution": float(contributions[i]),
        })

    # Cross-check against sklearn so the displayed arithmetic always matches
    # the probability used for ranking.
    sklearn_probability = float(model.logistic.predict_proba(row[FEATURES])[:, 1][0])

    return {
        "intercept": intercept,
        "logit": logit,
        "probability": probability,
        "sklearn_probability": sklearn_probability,
        "features": features,
        "formula": "P(Y=1)=1/(1+e^(-z))",
        "logit_formula": "z=β0+β1Z1+β2Z2+β3Z3+β4Z4",
        "standardization_formula": "Zj=(Xj-μj)/σj",
    }


def explain_creator(model: ContextModel, scored: pd.DataFrame, handle: str, decision_threshold: float = 0.50) -> dict:
    if scored.empty:
        raise ValueError("No creators available for this filter.")

    match = scored[scored["handle"] == handle]
    if match.empty:
        raise ValueError(f"Creator not found in the current result set: {handle}")

    row = match.iloc[[0]]
    values = row[FEATURES]
    path = model.tree.decision_path(values).indices.tolist()
    leaf_id = int(model.tree.apply(values)[0])

    path_steps = []
    tree_ = model.tree.tree_
    for node_id in path:
        if tree_.children_left[node_id] == tree_.children_right[node_id]:
            continue
        feature = FEATURES[int(tree_.feature[node_id])]
        threshold = float(tree_.threshold[node_id])
        actual = float(row.iloc[0][feature])
        went_left = actual <= threshold
        path_steps.append({
            "node_id": int(node_id),
            "feature": feature,
            "feature_label": FEATURE_LABELS[feature],
            "operator": "≤" if went_left else ">",
            "threshold": threshold,
            "threshold_label": _format_threshold(feature, threshold),
            "actual": actual,
            "actual_label": _format_actual(feature, actual),
        })

    selected = creator_record(row.iloc[0])
    tree_classes = list(model.tree.classes_)
    tree_proba = model.tree.predict_proba(values)[0]
    positive_index = tree_classes.index(1) if 1 in tree_classes else len(tree_classes) - 1
    tree_positive_probability = float(tree_proba[positive_index])
    decision_threshold = float(np.clip(float(decision_threshold), 0.0, 1.0))
    tree_prediction = int(tree_positive_probability >= decision_threshold)
    selected.update({
        "leaf_id": leaf_id,
        "decision_path": path_steps,
        "path_node_ids": [int(x) for x in path],
        "tree_positive_probability": tree_positive_probability,
        "tree_prediction": tree_prediction,
        "decision_threshold": decision_threshold,
    })
    return {
        "selected": selected,
        "tree": export_tree(model.tree, decision_threshold),
        "logistic_explanation": logistic_explanation(model, row),
    }


def fit_method_summary(brand_weight: float = DEFAULT_BRAND_WEIGHT) -> dict:
    brand_weight = _normalize_brand_weight(brand_weight)
    campaign_weight = 1.0 - brand_weight
    return {
        "brand_weight": brand_weight,
        "campaign_weight": campaign_weight,
        "label": f"{int(round(brand_weight*100))}% Brand + {int(round(campaign_weight*100))}% Campaign",
        "scale": FIT_SCALE_LABEL,
    }


def data_quality_summary() -> dict:
    df = DATA.creators
    return {
        "rows": int(len(df)),
        "repaired_rows": int(DATA.repaired_rows),
        "platform_counts": {
            key.title(): int(value)
            for key, value in df["platform"].value_counts().to_dict().items()
        },
        "niche_counts": {
            str(key): int(value)
            for key, value in df["niche"].value_counts().head(15).to_dict().items()
        },
        "country_counts": {
            key: int(value) for key, value in df["country_group"].value_counts().to_dict().items()
        },
        "engagement_imputed": int(df["engagement_imputed"].sum()),
        "verified_imputed": int(df["verified_imputed"].sum()),
        "fake_followers_nonzero": int((df["fake_followers_pct"].fillna(0) != 0).sum()),
        "note": "Missing public engagement/verification fields are imputed for modeling; fake_followers_pct is excluded from the model.",
    }
