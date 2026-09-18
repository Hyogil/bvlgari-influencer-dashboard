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

# Campaign-conditioned direct features.
# The selected campaign chooses the directly relevant content-ratio column(s).
# No manual weighted Content Fit, History Score, Brand Score, Campaign Score,
# Profile Fit, or Scenario Score is used.
CAMPAIGN_FEATURES: Dict[str, List[str]] = {
    "Luxury / Fashion": ["luxury_post_ratio_sim", "fashion_apparel_post_ratio_sim"],
    "Jewelry": ["jewelry_post_ratio_sim"],
    "Beauty": ["beauty_post_ratio_sim"],
    "Lifestyle": ["lifestyle_post_ratio_sim"],
}

BASE_MODEL_FEATURES = [
    "past_campaign_count_sim",
    "past_campaign_success_rate_sim",
    "engagement_rate",
    "followers_log",
]

FEATURE_LABELS = {
    "luxury_post_ratio_sim": "Luxury Post Ratio",
    "jewelry_post_ratio_sim": "Jewelry Post Ratio",
    "fashion_apparel_post_ratio_sim": "Fashion / Apparel Post Ratio",
    "beauty_post_ratio_sim": "Beauty Post Ratio",
    "lifestyle_post_ratio_sim": "Lifestyle Post Ratio",
    "past_campaign_count_sim": "Past Campaign Count",
    "past_campaign_success_rate_sim": "Past Campaign Success Rate",
    "engagement_rate": "Engagement Rate",
    "followers_log": "Follower Reach (log)",
}

DEFAULT_CONTROLS = {
    "min_followers": 0.0,
    "min_engagement": 0.0,
    "suitable_threshold": 0.50,
    "tree_depth": 4,
}


@dataclass
class DataBundle:
    creators: pd.DataFrame
    repaired_rows: int
    raw_encoding: str


@dataclass
class ContextModel:
    campaign: str
    country: str
    tree_depth: int
    features: List[str]
    scored: pd.DataFrame
    logistic: Pipeline
    tree: DecisionTreeClassifier
    target_positive_rate: float
    training_size: int


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, sheet_name=DATA_SHEET, engine="openpyxl")
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def _repair_shifted_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
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
        "fashion_apparel_post_ratio_sim", "beauty_post_ratio_sim",
        "lifestyle_post_ratio_sim", "past_campaign_count_sim",
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
        "fashion_apparel_post_ratio_sim", "beauty_post_ratio_sim",
        "lifestyle_post_ratio_sim", "past_campaign_count_sim",
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
        "fashion_apparel_post_ratio_sim", "beauty_post_ratio_sim",
        "lifestyle_post_ratio_sim", "past_campaign_success_rate_sim",
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


def filter_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    if country not in VALID_COUNTRIES:
        raise ValueError(f"Unknown country filter: {country}")
    if country == "All":
        return df.copy()
    return df[df["country_group"] == country].copy()


def model_features_for_campaign(campaign: str) -> List[str]:
    if campaign not in CAMPAIGN_FEATURES:
        raise ValueError(f"Unknown campaign: {campaign}")
    return [*CAMPAIGN_FEATURES[campaign], *BASE_MODEL_FEATURES]


@lru_cache(maxsize=128)
def train_context(
    campaign: str = "Luxury / Fashion",
    country: str = "All",
    tree_depth: int = 4,
) -> ContextModel:
    if campaign not in CAMPAIGN_FEATURES:
        raise ValueError(f"Unknown campaign: {campaign}")
    if country not in VALID_COUNTRIES:
        raise ValueError(f"Unknown country filter: {country}")
    if not 1 <= int(tree_depth) <= 8:
        raise ValueError("Tree depth must be between 1 and 8.")

    depth = int(tree_depth)
    features = model_features_for_campaign(campaign)
    country_df = filter_country(DATA.creators, country)
    if len(country_df) < 20:
        raise ValueError(f"Not enough creators in country filter: {country}")

    df = country_df.copy()
    y = df[TARGET_COLUMN].astype(int)
    if y.nunique() < 2:
        raise ValueError(f"Target has only one class in country filter: {country}")
    X = df[features]

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
        campaign=campaign,
        country=country,
        tree_depth=depth,
        features=features,
        scored=scored,
        logistic=logistic,
        tree=tree,
        target_positive_rate=float(y.mean()),
        training_size=int(len(df)),
    )


def score_creators(
    campaign: str = "Luxury / Fashion",
    platform: str = "All",
    country: str = "All",
    min_followers: float = 0.0,
    min_engagement: float = 0.0,
    suitable_threshold: float = 0.50,
    tree_depth: int = 4,
) -> tuple[ContextModel, pd.DataFrame, dict]:
    model = train_context(campaign, country, int(tree_depth))
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

    scored = scored.sort_values(
        ["selection_probability", "engagement_rate", "followers"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)

    control_summary = {
        "campaign": campaign,
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
        "luxury_post_ratio_sim", "jewelry_post_ratio_sim",
        "fashion_apparel_post_ratio_sim", "beauty_post_ratio_sim",
        "lifestyle_post_ratio_sim", "past_campaign_success_rate_sim",
    }:
        return f"{threshold * 100:.1f}%"
    if feature == "past_campaign_count_sim":
        return f"{threshold:.1f} campaigns"
    return f"{threshold:.2f}"


def _format_actual(feature: str, value: float) -> str:
    if feature == "past_campaign_count_sim":
        return f"{int(round(value))} campaigns"
    return _format_threshold(feature, value)


def export_tree(model: ContextModel, suitable_threshold: float = 0.50) -> dict:
    tree_ = model.tree.tree_
    threshold_for_label = float(np.clip(suitable_threshold, 0.0, 1.0))
    features = model.features

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

        feature = features[int(tree_.feature[node_id])]
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


def _campaign_content_values(row: pd.Series, campaign: str) -> List[dict]:
    return [
        {
            "feature": feature,
            "label": FEATURE_LABELS[feature],
            "ratio": float(row[feature]),
        }
        for feature in CAMPAIGN_FEATURES[campaign]
    ]


def creator_record(row: pd.Series, campaign: str) -> dict:
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
        "luxury_post_ratio": float(row["luxury_post_ratio_sim"]),
        "jewelry_post_ratio": float(row["jewelry_post_ratio_sim"]),
        "fashion_post_ratio": float(row["fashion_apparel_post_ratio_sim"]),
        "beauty_post_ratio": float(row["beauty_post_ratio_sim"]),
        "lifestyle_post_ratio": float(row["lifestyle_post_ratio_sim"]),
        "campaign": campaign,
        "campaign_content": _campaign_content_values(row, campaign),
        "past_campaign_count": int(round(float(row["past_campaign_count_sim"]))),
        "past_campaign_success_rate": float(row["past_campaign_success_rate_sim"]),
        "selection_probability": float(row["selection_probability"]),
        "tree_probability": float(row.get("tree_probability", np.nan)),
        "tree_suitable": bool(row.get("tree_suitable", False)),
        "rank": int(row["rank"]),
    }


def top_records(scored: pd.DataFrame, campaign: str, n: int = 10) -> List[dict]:
    return [creator_record(row, campaign) for _, row in scored.head(n).iterrows()]


def logistic_explanation(model: ContextModel, row: pd.DataFrame) -> dict:
    scaler: StandardScaler = model.logistic.named_steps["scale"]
    classifier: LogisticRegression = model.logistic.named_steps["model"]
    features = model.features

    raw = row[features].iloc[0].astype(float).to_numpy()
    means = scaler.mean_.astype(float)
    scales = scaler.scale_.astype(float)
    standardized = (raw - means) / scales
    coefficients = classifier.coef_[0].astype(float)
    intercept = float(classifier.intercept_[0])
    contributions = coefficients * standardized
    logit = float(intercept + contributions.sum())
    probability = float(1.0 / (1.0 + np.exp(-logit)))

    feature_rows = []
    for i, feature in enumerate(features):
        feature_rows.append({
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

    sklearn_probability = float(model.logistic.predict_proba(row[features])[:, 1][0])
    return {
        "intercept": intercept,
        "logit": logit,
        "probability": probability,
        "sklearn_probability": sklearn_probability,
        "features": feature_rows,
        "feature_count": len(features),
        "campaign": model.campaign,
        "formula": "P(Y=1)=1/(1+e^(-z))",
        "logit_formula": f"z=β0+ΣβjZj ({len(features)} features)",
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
    features = model.features
    values = row[features]
    path = model.tree.decision_path(values).indices.tolist()
    leaf_id = int(model.tree.apply(values)[0])

    path_steps = []
    tree_ = model.tree.tree_
    for node_id in path:
        if tree_.children_left[node_id] == tree_.children_right[node_id]:
            continue
        feature = features[int(tree_.feature[node_id])]
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

    selected = creator_record(row.iloc[0], model.campaign)
    selected.update({
        "leaf_id": leaf_id,
        "decision_path": path_steps,
        "path_node_ids": [int(x) for x in path],
        "tree_suitable": bool(float(row.iloc[0]["tree_probability"]) >= float(suitable_threshold)),
    })
    return {
        "selected": selected,
        "tree": export_tree(model, suitable_threshold),
        "logistic_explanation": logistic_explanation(model, row),
    }


def fit_method_summary(campaign: str = "Luxury / Fashion") -> dict:
    features = model_features_for_campaign(campaign)
    campaign_features = CAMPAIGN_FEATURES[campaign]
    content_labels = [FEATURE_LABELS[f] for f in campaign_features]
    return {
        "label": f"Campaign-conditioned Logistic Regression · {campaign}",
        "campaign": campaign,
        "feature_count": len(features),
        "features": [FEATURE_LABELS[f] for f in features],
        "campaign_content_features": content_labels,
        "weighting_note": (
            "The selected campaign determines which direct post-ratio feature(s) enter the model. "
            "There is no manually weighted fit score. Logistic Regression learns standardized coefficients "
            "for the selected content ratio(s), campaign history, engagement, and reach."
        ),
        "target": TARGET_COLUMN,
        "target_note": "Synthetic historical outcome for prototype use only",
    }


def ranking_summary(campaign: str = "Luxury / Fashion") -> dict:
    labels = [FEATURE_LABELS[f] for f in CAMPAIGN_FEATURES[campaign]]
    return {
        "label": f"Ranking = {campaign}-conditioned Logistic Regression probability",
        "campaign": campaign,
        "campaign_content_features": labels,
        "note": (
            f"Selecting {campaign} changes the model feature set to use the directly relevant content ratio(s): "
            f"{', '.join(labels)}. Final ranking uses only that campaign-conditioned Logistic Regression probability. "
            "No heuristic weighting or Scenario Score is blended into ranking."
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
        "campaigns": list(CAMPAIGN_FEATURES.keys()),
        "note": (
            "Campaign-conditioned models use direct content-ratio variables with no heuristic Brand/Campaign scores "
            "and no manually weighted composite scores. Added *_sim variables and the target remain synthetic prototype data."
        ),
    }
