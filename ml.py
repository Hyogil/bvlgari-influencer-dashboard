from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creators_enriched_synthetic_v1.xlsx"
DATA_SHEET = "Creators_Enriched"
TARGET_COLUMN = "simulated_campaign_success_target"

VALID_PLATFORMS = {"youtube", "instagram", "tiktok"}
VALID_COUNTRIES = {"All", "KR", "Global"}
KOREA_LOCATION_KEYS = {"kr", "south korea", "korea", "republic of korea", "korea, republic of"}

# Direct 9-feature model.
# No manually weighted composite Content Fit, Campaign History Score, or Profile Fit is used.
# Each source variable enters Logistic Regression separately, so its relative contribution
# is learned through the standardized regression coefficients.
FEATURES = [
    "brand_score",
    "campaign_score",
    "luxury_post_ratio_sim",
    "jewelry_post_ratio_sim",
    "fashion_apparel_post_ratio_sim",
    "past_campaign_count_sim",
    "past_campaign_success_rate_sim",
    "engagement_rate",
    "followers_log",
]

FEATURE_LABELS = {
    "brand_score": "Brand Score",
    "campaign_score": "Campaign Score",
    "luxury_post_ratio_sim": "Luxury Post Ratio",
    "jewelry_post_ratio_sim": "Jewelry Post Ratio",
    "fashion_apparel_post_ratio_sim": "Fashion / Apparel Post Ratio",
    "past_campaign_count_sim": "Past Campaign Count",
    "past_campaign_success_rate_sim": "Past Campaign Success Rate",
    "engagement_rate": "Engagement Rate",
    "followers_log": "Follower Reach (log)",
}

FIT_SCALE_LABEL = "Brand/Campaign heuristic score scale: 1.00 / 0.75 / 0.50 / 0.25 / 0.00"

DEFAULT_CONTROLS = {
    "min_followers": 0.0,
    "min_engagement": 0.0,
    "suitable_threshold": 0.50,
    "tree_depth": 4,
}

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
    # Campaign Fit is deliberately different from Brand Fit.
    # Brand Fit represents long-term affinity with the selected brand, while
    # Campaign Fit represents short-term activation suitability for the campaign.
    # These are business heuristic assumptions on the same 5-level ordinal scale;
    # they are NOT learned Logistic Regression coefficients.
    "Luxury / Fashion": {
        "fashion": 1.00,
        "fashion_&_style": 1.00,
        "lifestyle": 0.75,
        "beauty": 0.50,
        "travel": 0.50,
        "music": 0.50,
        "fitness": 0.25,
        "comedy": 0.25,
        "food": 0.25,
        "business": 0.00,
        "tech": 0.00,
        "gaming": 0.00,
        "education": 0.00,
        "finance": 0.00,
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
    tree_depth: int
    scored: pd.DataFrame
    logistic: Pipeline
    tree: DecisionTreeClassifier
    target_positive_rate: float
    training_size: int


def _read_dataset(path: Path) -> pd.DataFrame:
    """Read the enriched Excel dataset used by the direct 9-feature model."""
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, sheet_name=DATA_SHEET, engine="openpyxl")
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def _repair_shifted_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Repair legacy rows whose platform token was appended to the name field."""
    out = df.copy()
    for col in ["platform", "niche", "followers", "avg_engagement_rate", "location", "verified", "fake_followers_pct"]:
        out[col] = out[col].astype(object)
    invalid = ~out["platform"].astype(str).str.lower().isin(VALID_PLATFORMS)
    repaired = 0

    for idx in out.index[invalid]:
        name = str(out.at[idx, "name"])
        match = re.search(r"(youtube|instagram|tiktok)\s*$", name, flags=re.I)
        if not match:
            continue
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
    raw = _read_dataset(DATA_PATH)
    df = raw.copy()

    if "data_repaired_flag" in df.columns:
        repaired = int(pd.to_numeric(df["data_repaired_flag"], errors="coerce").fillna(0).sum())
    else:
        df, repaired = _repair_shifted_rows(df)

    required = {
        "handle", "name", "platform", "niche", "followers",
        "avg_engagement_rate", "location", "verified", "fake_followers_pct",
        "luxury_post_ratio_sim", "jewelry_post_ratio_sim",
        "fashion_apparel_post_ratio_sim", "past_campaign_count_sim",
        "past_campaign_success_rate_sim", TARGET_COLUMN,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df["handle"] = df["handle"].astype(str).str.strip()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["platform"] = df["platform"].astype(str).str.strip().str.lower()
    df["niche"] = df["niche"].fillna("unknown").astype(str).str.strip().str.lower()

    numeric_cols = [
        "followers", "avg_engagement_rate", "fake_followers_pct",
        "luxury_post_ratio_sim", "jewelry_post_ratio_sim",
        "fashion_apparel_post_ratio_sim", "past_campaign_count_sim",
        "past_campaign_success_rate_sim", TARGET_COLUMN,
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    verified_text = df["verified"].fillna("").astype(str).str.upper().str.strip()
    true_values = {"TRUE", "1", "YES", "Y"}
    false_values = {"FALSE", "0", "NO", "N"}
    verified_raw = pd.Series(np.nan, index=df.index, dtype=float)
    verified_raw[verified_text.isin(true_values)] = 1.0
    verified_raw[verified_text.isin(false_values)] = 0.0
    df["verified_known"] = verified_raw.notna()

    df = df[df["platform"].isin(VALID_PLATFORMS)].copy()
    df = df.dropna(subset=["followers", TARGET_COLUMN]).copy()
    df["followers"] = df["followers"].clip(lower=0)
    df["avg_engagement_rate"] = df["avg_engagement_rate"].clip(lower=0, upper=50)

    location_key = df["location"].fillna("").astype(str).str.strip().str.lower()
    df["country_group"] = np.where(location_key.isin(KOREA_LOCATION_KEYS), "KR", "Global")

    df["engagement_imputed"] = df["avg_engagement_rate"].isna()
    platform_engagement_median = df.groupby("platform")["avg_engagement_rate"].transform("median")
    overall_engagement_median = float(df["avg_engagement_rate"].median())
    df["engagement_rate"] = (
        df["avg_engagement_rate"]
        .fillna(platform_engagement_median)
        .fillna(overall_engagement_median)
        .astype(float)
    )

    known_verified = verified_raw.reindex(df.index)
    tmp = pd.DataFrame({"platform": df["platform"], "value": known_verified})
    medians = tmp.groupby("platform")["value"].transform("median")
    overall_verified = float(known_verified.median()) if known_verified.notna().any() else 0.0
    df["verified_imputed"] = known_verified.isna()
    df["verified_flag"] = known_verified.fillna(medians).fillna(overall_verified).round().astype(int)

    df["followers_log"] = np.log1p(df["followers"].astype(float))
    for col in [
        "luxury_post_ratio_sim", "jewelry_post_ratio_sim",
        "fashion_apparel_post_ratio_sim", "past_campaign_success_rate_sim",
    ]:
        df[col] = df[col].fillna(0).clip(lower=0, upper=1).astype(float)
    df["past_campaign_count_sim"] = df["past_campaign_count_sim"].fillna(0).clip(lower=0).astype(float)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(0).astype(int).clip(lower=0, upper=1)
    df["fake_followers_pct"] = df["fake_followers_pct"].fillna(0).astype(float)

    return DataBundle(
        creators=df.reset_index(drop=True),
        repaired_rows=repaired,
        raw_encoding="Excel workbook / openpyxl",
    )


DATA = load_data()


def _profile_score(niche: str, profile: Dict[str, float]) -> float:
    key = str(niche).lower().strip()
    if key in profile:
        return float(profile[key])
    if "fashion" in key:
        return float(profile.get("fashion", 0.75))
    return 0.00


def compute_context_scores(niche: str, brand: str, campaign: str) -> tuple[float, float]:
    """Return independent (brand_score, campaign_score) without a blending weight."""
    brand_score = _profile_score(niche, BRAND_PROFILES[brand])
    campaign_score = _profile_score(niche, CAMPAIGN_PROFILES[campaign])
    return float(brand_score), float(campaign_score)


def enrich_context(df: pd.DataFrame, brand: str, campaign: str) -> pd.DataFrame:
    out = df.copy()
    scores = out["niche"].map(lambda n: compute_context_scores(n, brand, campaign))
    out["brand_score"] = scores.map(lambda x: x[0])
    out["campaign_score"] = scores.map(lambda x: x[1])
    return out


def filter_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    if country not in VALID_COUNTRIES:
        raise ValueError(f"Unknown country filter: {country}")
    if country == "All":
        return df.copy()
    return df[df["country_group"] == country].copy()



@lru_cache(maxsize=192)
def train_context(
    brand: str,
    campaign: str,
    country: str = "All",
    tree_depth: int = 4,
) -> ContextModel:
    if brand not in BRAND_PROFILES:
        raise ValueError(f"Unknown brand: {brand}")
    if campaign not in CAMPAIGN_PROFILES:
        raise ValueError(f"Unknown campaign: {campaign}")
    if country not in VALID_COUNTRIES:
        raise ValueError(f"Unknown country filter: {country}")
    if not 1 <= int(tree_depth) <= 8:
        raise ValueError("Tree depth must be between 1 and 8.")

    depth = int(tree_depth)
    country_df = filter_country(DATA.creators, country)
    if len(country_df) < 20:
        raise ValueError(f"Not enough creators in country filter: {country}")

    df = enrich_context(country_df, brand, campaign)
    y = df[TARGET_COLUMN].astype(int)
    if y.nunique() < 2:
        raise ValueError(f"Target has only one class in country filter: {country}")
    X = df[FEATURES]

    logistic = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    logistic.fit(X, y)

    min_leaf = max(8, min(120, int(round(len(df) * 0.08))))
    tree = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(X, y)

    scored = df.copy()
    scored["selection_probability"] = logistic.predict_proba(X)[:, 1]
    scored["tree_probability"] = tree.predict_proba(X)[:, 1]
    scored[TARGET_COLUMN] = y

    return ContextModel(
        brand=brand,
        campaign=campaign,
        country=country,
        tree_depth=depth,
        scored=scored,
        logistic=logistic,
        tree=tree,
        target_positive_rate=float(y.mean()),
        training_size=int(len(df)),
    )


def score_creators(
    brand: str,
    campaign: str,
    platform: str = "All",
    country: str = "All",
    min_followers: float = 0.0,
    min_engagement: float = 0.0,
    suitable_threshold: float = 0.50,
    tree_depth: int = 4,
) -> tuple[ContextModel, pd.DataFrame, dict]:
    model = train_context(brand, campaign, country, int(tree_depth))
    scored = model.scored.copy()

    if platform and platform != "All":
        platform_key = platform.strip().lower()
        scored = scored[scored["platform"] == platform_key].copy()

    scored = scored[
        (scored["followers"] >= float(min_followers))
        & (scored["engagement_rate"] >= float(min_engagement))
    ].copy()

    threshold = float(np.clip(suitable_threshold, 0.0, 1.0))
    scored["tree_suitable"] = scored["tree_probability"] >= threshold

    # Final ranking is based solely on Logistic Regression probability.
    # Engagement and Followers are deterministic tie-breakers only.
    scored = scored.sort_values(
        ["selection_probability", "engagement_rate", "followers"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)

    control_summary = {
        "min_followers": float(min_followers),
        "min_engagement": float(min_engagement),
        "suitable_threshold": threshold,
        "tree_depth": int(tree_depth),
    }
    return model, scored, control_summary


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
    if feature == "followers_log":
        followers = max(0.0, float(np.expm1(threshold)))
        return f"{_compact_count(followers)} (log {threshold:.2f})"
    if feature in {
        "luxury_post_ratio_sim",
        "jewelry_post_ratio_sim",
        "fashion_apparel_post_ratio_sim",
        "past_campaign_success_rate_sim",
    }:
        return f"{threshold * 100:.1f}%"
    if feature in {"brand_score", "campaign_score"}:
        return f"{threshold:.3f}"
    if feature == "past_campaign_count_sim":
        return f"{threshold:.1f} campaigns"
    return f"{threshold:.2f}"


def _format_actual(feature: str, value: float) -> str:
    if feature == "past_campaign_count_sim":
        return f"{int(round(value))} campaigns"
    return _format_threshold(feature, value)


def export_tree(model: DecisionTreeClassifier, suitable_threshold: float = 0.50) -> dict:
    tree_ = model.tree_
    threshold_for_label = float(np.clip(suitable_threshold, 0.0, 1.0))

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
            item["label"] = "Suitable" if positive_prob >= threshold_for_label else "Not Suitable"
            return item

        feature = FEATURES[int(tree_.feature[node_id])]
        split_threshold = float(tree_.threshold[node_id])
        item.update({
            "feature": feature,
            "feature_label": FEATURE_LABELS[feature],
            "threshold": split_threshold,
            "threshold_label": _format_threshold(feature, split_threshold),
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
        "brand_score": float(row["brand_score"]),
        "campaign_score": float(row["campaign_score"]),
        "luxury_post_ratio": float(row["luxury_post_ratio_sim"]),
        "jewelry_post_ratio": float(row["jewelry_post_ratio_sim"]),
        "fashion_post_ratio": float(row["fashion_apparel_post_ratio_sim"]),
        "past_campaign_count": int(round(float(row["past_campaign_count_sim"]))),
        "past_campaign_success_rate": float(row["past_campaign_success_rate_sim"]),
        "selection_probability": float(row["selection_probability"]),
        "tree_probability": float(row.get("tree_probability", np.nan)),
        "tree_suitable": bool(row.get("tree_suitable", False)),
        "rank": int(row["rank"]),
    }


def top_records(scored: pd.DataFrame, n: int = 10) -> List[dict]:
    return [creator_record(row) for _, row in scored.head(n).iterrows()]


def logistic_explanation(model: ContextModel, row: pd.DataFrame) -> dict:
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

    sklearn_probability = float(model.logistic.predict_proba(row[FEATURES])[:, 1][0])
    return {
        "intercept": intercept,
        "logit": logit,
        "probability": probability,
        "sklearn_probability": sklearn_probability,
        "features": features,
        "formula": "P(Y=1)=1/(1+e^(-z))",
        "logit_formula": "z=β0+ΣβjZj (9 features)",
        "standardization_formula": "Zj=(Xj-μj)/σj",
    }


def explain_creator(
    model: ContextModel,
    scored: pd.DataFrame,
    handle: str,
    suitable_threshold: float = 0.50,
) -> dict:
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
        split_threshold = float(tree_.threshold[node_id])
        actual = float(row.iloc[0][feature])
        went_left = actual <= split_threshold
        path_steps.append({
            "node_id": int(node_id),
            "feature": feature,
            "feature_label": FEATURE_LABELS[feature],
            "operator": "≤" if went_left else ">",
            "threshold": split_threshold,
            "threshold_label": _format_threshold(feature, split_threshold),
            "actual": actual,
            "actual_label": _format_actual(feature, actual),
        })

    selected = creator_record(row.iloc[0])
    selected.update({
        "leaf_id": leaf_id,
        "decision_path": path_steps,
        "path_node_ids": [int(x) for x in path],
        "tree_suitable": bool(float(row.iloc[0]["tree_probability"]) >= float(suitable_threshold)),
    })
    return {
        "selected": selected,
        "tree": export_tree(model.tree, suitable_threshold),
        "logistic_explanation": logistic_explanation(model, row),
    }


def fit_method_summary() -> dict:
    return {
        "label": "9-feature direct Logistic Regression · no manually weighted composite scores",
        "scale": FIT_SCALE_LABEL,
        "feature_count": len(FEATURES),
        "features": [FEATURE_LABELS[f] for f in FEATURES],
        "weighting_note": (
            "Brand Score, Campaign Score, Luxury/Jewelry/Fashion ratios, Past Campaign Count, "
            "Past Campaign Success Rate, Engagement, and log Followers enter the model separately. "
            "Logistic Regression learns their standardized coefficients."
        ),
        "target": TARGET_COLUMN,
        "target_note": "Synthetic historical outcome for prototype use only",
    }


def ranking_summary() -> dict:
    return {
        "label": "Ranking = Logistic Regression predicted probability",
        "note": (
            "Final candidate ranking uses only the 9-feature Logistic Regression probability. "
            "No Scenario Score, Content Fit weighted sum, Campaign History weighted sum, or "
            "Brand↔Campaign blend is used."
        ),
    }


def data_quality_summary() -> dict:
    df = DATA.creators
    return {
        "rows": int(len(df)),
        "repaired_rows": int(DATA.repaired_rows),
        "platform_counts": {key.title(): int(value) for key, value in df["platform"].value_counts().to_dict().items()},
        "niche_counts": {str(key): int(value) for key, value in df["niche"].value_counts().head(15).to_dict().items()},
        "country_counts": {key: int(value) for key, value in df["country_group"].value_counts().to_dict().items()},
        "engagement_imputed": int(df["engagement_imputed"].sum()),
        "verified_imputed": int(df["verified_imputed"].sum()),
        "fake_followers_nonzero": int((df["fake_followers_pct"].fillna(0) != 0).sum()),
        "note": "Model uses 9 direct features with no manually weighted composite Content Fit, Campaign History Score, or Profile Fit. Added *_sim variables and the target remain synthetic prototype data.",
    }
