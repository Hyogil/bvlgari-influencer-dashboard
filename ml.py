from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence
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
TARGET_COLUMN = "simulated_campaign_success_target"

# Curated Enhanced Model = 8 features.
CORE_FEATURES = ["brand_fit", "engagement_rate", "followers_log"]
CONTENT_HISTORY_FEATURES = [
    "luxury_post_ratio_sim",
    "jewelry_post_ratio_sim",
    "fashion_apparel_post_ratio_sim",
]
CAMPAIGN_HISTORY_FEATURES = [
    "past_campaign_count_sim",
    "past_campaign_success_rate_sim",
]
CURATED_FEATURES = CORE_FEATURES + CONTENT_HISTORY_FEATURES + CAMPAIGN_HISTORY_FEATURES
BASELINE_FEATURES = ["brand_fit", "engagement_rate", "followers_m", "verified_flag"]

FEATURE_LABELS = {
    "brand_fit": "Brand Fit Score",
    "engagement_rate": "Engagement Rate",
    "followers_log": "Followers (log10)",
    "followers_m": "Followers",
    "verified_flag": "Verified Creator",
    "luxury_post_ratio_sim": "Luxury Content Ratio",
    "jewelry_post_ratio_sim": "Jewelry Content Ratio",
    "fashion_apparel_post_ratio_sim": "Fashion / Apparel Ratio",
    "past_campaign_count_sim": "Past Campaign Count",
    "past_campaign_success_rate_sim": "Past Campaign Success Rate",
}

DEFAULT_BRAND_WEIGHT = 0.50
FIT_SCALE_LABEL = "Ordinal 5-level heuristic scale (1.00 / 0.75 / 0.50 / 0.25 / 0.00)"

BRAND_PROFILES: Dict[str, Dict[str, float]] = {
    "BVLGARI": {
        "fashion": 1.00, "fashion_&_style": 1.00, "beauty": 0.75,
        "lifestyle": 0.50, "travel": 0.25, "music": 0.25,
        "fitness": 0.00, "food": 0.00, "business": 0.00, "tech": 0.00,
        "gaming": 0.00, "education": 0.00, "finance": 0.00, "comedy": 0.00,
    },
    "Gucci": {
        "fashion": 1.00, "fashion_&_style": 1.00, "beauty": 0.75,
        "lifestyle": 0.75, "music": 0.50, "travel": 0.50, "comedy": 0.25,
        "fitness": 0.25, "food": 0.25, "business": 0.00, "tech": 0.00,
        "gaming": 0.00, "education": 0.00, "finance": 0.00,
    },
    "Dior": {
        "fashion": 1.00, "fashion_&_style": 1.00, "beauty": 1.00,
        "lifestyle": 0.75, "travel": 0.50, "music": 0.25, "fitness": 0.25,
        "food": 0.00, "business": 0.00, "tech": 0.00, "gaming": 0.00,
        "education": 0.00, "finance": 0.00, "comedy": 0.00,
    },
    "Nike": {
        "fitness": 1.00, "lifestyle": 0.75, "fashion": 0.50,
        "fashion_&_style": 0.50, "travel": 0.50, "music": 0.25,
        "gaming": 0.25, "comedy": 0.25, "food": 0.25, "business": 0.00,
        "tech": 0.25, "education": 0.00, "finance": 0.00, "beauty": 0.25,
    },
}

CAMPAIGN_PROFILES: Dict[str, Dict[str, float]] = {
    "Luxury / Fashion": {
        "fashion": 1.00, "fashion_&_style": 1.00, "beauty": 0.75,
        "lifestyle": 0.50, "travel": 0.25, "music": 0.25, "fitness": 0.00,
        "food": 0.00, "business": 0.00, "tech": 0.00, "gaming": 0.00,
        "education": 0.00, "finance": 0.00, "comedy": 0.00,
    },
    "Jewelry": {
        "fashion": 1.00, "fashion_&_style": 1.00, "beauty": 0.75,
        "lifestyle": 0.75, "travel": 0.25, "music": 0.25, "finance": 0.25,
        "fitness": 0.00, "food": 0.00, "business": 0.00, "tech": 0.00,
        "gaming": 0.00, "education": 0.00, "comedy": 0.00,
    },
    "Beauty": {
        "beauty": 1.00, "fashion": 0.75, "fashion_&_style": 0.75,
        "lifestyle": 0.75, "travel": 0.25, "fitness": 0.25, "music": 0.25,
        "food": 0.00, "business": 0.00, "tech": 0.00, "gaming": 0.00,
        "education": 0.00, "finance": 0.00, "comedy": 0.00,
    },
    "Lifestyle": {
        "lifestyle": 1.00, "travel": 0.75, "fashion": 0.50,
        "fashion_&_style": 0.50, "beauty": 0.50, "fitness": 0.50,
        "food": 0.50, "music": 0.25, "comedy": 0.25, "business": 0.25,
        "tech": 0.25, "gaming": 0.25, "education": 0.25, "finance": 0.00,
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
    baseline_logistic: Pipeline
    tree: DecisionTreeClassifier
    feature_names: List[str]
    target_positive_rate: float
    training_size: int


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def _repair_shifted_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fallback repair for legacy malformed rows. The enriched v1 file already stores a repair flag."""
    out = df.copy()
    if "data_repaired_flag" not in out.columns:
        out["data_repaired_flag"] = 0
    already_repaired = int(pd.to_numeric(out["data_repaired_flag"], errors="coerce").fillna(0).sum())

    needed = ["platform", "niche", "followers", "avg_engagement_rate", "location", "verified", "fake_followers_pct"]
    for col in needed:
        if col in out.columns:
            out[col] = out[col].astype(object)

    invalid = ~out["platform"].astype(str).str.lower().isin(VALID_PLATFORMS)
    additional = 0
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
        out.at[idx, "name"] = name[:match.start()].rstrip(" ?|-/")
        out.at[idx, "platform"] = match.group(1).lower()
        out.at[idx, "niche"] = old_platform
        out.at[idx, "followers"] = old_niche
        out.at[idx, "avg_engagement_rate"] = old_followers
        out.at[idx, "location"] = old_engagement
        out.at[idx, "verified"] = old_location
        out.at[idx, "fake_followers_pct"] = old_verified
        out.at[idx, "data_repaired_flag"] = 1
        additional += 1
    return out, already_repaired + additional


def load_data() -> DataBundle:
    raw = _read_csv(DATA_PATH)
    df, repaired = _repair_shifted_rows(raw)

    required = {
        "handle", "name", "platform", "niche", "followers", "avg_engagement_rate",
        "location", "verified", TARGET_COLUMN,
        *CONTENT_HISTORY_FEATURES, *CAMPAIGN_HISTORY_FEATURES,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Enhanced model dataset is missing columns: {', '.join(missing)}")

    df["handle"] = df["handle"].astype(str).str.strip()
    df["name"] = df["name"].fillna("").astype(str).str.strip()
    df["platform"] = df["platform"].astype(str).str.strip().str.lower()
    df["niche"] = df["niche"].fillna("unknown").astype(str).str.strip().str.lower()
    df["followers"] = pd.to_numeric(df["followers"], errors="coerce")
    df["avg_engagement_rate"] = pd.to_numeric(df["avg_engagement_rate"], errors="coerce")
    if "fake_followers_pct" in df.columns:
        df["fake_followers_pct"] = pd.to_numeric(df["fake_followers_pct"], errors="coerce")
    else:
        df["fake_followers_pct"] = 0.0

    verified_text = df["verified"].fillna("").astype(str).str.upper().str.strip()
    true_values = {"TRUE", "1", "YES", "Y"}
    false_values = {"FALSE", "0", "NO", "N"}
    verified_raw = pd.Series(np.nan, index=df.index, dtype=float)
    verified_raw[verified_text.isin(true_values)] = 1.0
    verified_raw[verified_text.isin(false_values)] = 0.0
    df["verified_known"] = verified_raw.notna()

    df = df[df["platform"].isin(VALID_PLATFORMS)].copy()
    df = df.dropna(subset=["followers"]).copy()
    df["followers"] = df["followers"].clip(lower=0)
    df["avg_engagement_rate"] = df["avg_engagement_rate"].clip(lower=0, upper=50)

    location_key = df["location"].fillna("").astype(str).str.strip().str.lower()
    df["country_group"] = np.where(location_key.isin(KOREA_LOCATION_KEYS), "KR", "Global")

    df["engagement_imputed"] = df["avg_engagement_rate"].isna()
    platform_engagement_median = df.groupby("platform")["avg_engagement_rate"].transform("median")
    overall_engagement_median = float(df["avg_engagement_rate"].median())
    df["engagement_rate"] = (
        df["avg_engagement_rate"].fillna(platform_engagement_median).fillna(overall_engagement_median).astype(float)
    )

    known_verified = verified_raw.reindex(df.index)
    tmp = pd.DataFrame({"platform": df["platform"], "value": known_verified})
    medians = tmp.groupby("platform")["value"].transform("median")
    overall_verified = float(known_verified.median()) if known_verified.notna().any() else 0.0
    df["verified_imputed"] = known_verified.isna()
    df["verified_flag"] = known_verified.fillna(medians).fillna(overall_verified).round().astype(int)

    df["followers_m"] = df["followers"].astype(float) / 1_000_000.0
    df["followers_log"] = np.log10(df["followers"].astype(float) + 1.0)

    for col in CONTENT_HISTORY_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lower=0, upper=1)
        df[col] = df[col].fillna(float(df[col].median()) if df[col].notna().any() else 0.0)
    df["past_campaign_count_sim"] = pd.to_numeric(df["past_campaign_count_sim"], errors="coerce").clip(lower=0).fillna(0)
    df["past_campaign_success_rate_sim"] = pd.to_numeric(df["past_campaign_success_rate_sim"], errors="coerce").clip(lower=0, upper=1).fillna(0)

    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df[target.isin([0, 1])].copy()
    df[TARGET_COLUMN] = target.loc[df.index].astype(int)

    return DataBundle(df.reset_index(drop=True), repaired, "utf-8-sig (legacy fallback supported)")


DATA = load_data()


def _profile_score(niche: str, profile: Dict[str, float]) -> float:
    key = str(niche).lower().strip()
    if key in profile:
        return float(profile[key])
    if "fashion" in key:
        return float(profile.get("fashion", 0.75))
    return 0.0


def _normalize_brand_weight(brand_weight: float) -> float:
    return float(np.clip(float(brand_weight), 0.0, 1.0))


def compute_brand_fit(niche: str, brand: str, campaign: str, brand_weight: float = DEFAULT_BRAND_WEIGHT) -> float:
    brand_weight = _normalize_brand_weight(brand_weight)
    campaign_weight = 1.0 - brand_weight
    brand_score = _profile_score(niche, BRAND_PROFILES[brand])
    campaign_score = _profile_score(niche, CAMPAIGN_PROFILES[campaign])
    return float(np.clip(brand_weight * brand_score + campaign_weight * campaign_score, 0.0, 1.0))


def enrich_context(df: pd.DataFrame, brand: str, campaign: str, brand_weight: float = DEFAULT_BRAND_WEIGHT) -> pd.DataFrame:
    out = df.copy()
    out["brand_fit"] = out["niche"].map(lambda n: compute_brand_fit(n, brand, campaign, brand_weight))
    return out


def active_features(use_content_history: bool = True, use_campaign_history: bool = True) -> List[str]:
    features = list(CORE_FEATURES)
    if use_content_history:
        features += CONTENT_HISTORY_FEATURES
    if use_campaign_history:
        features += CAMPAIGN_HISTORY_FEATURES
    return features


def filter_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    if country not in VALID_COUNTRIES:
        raise ValueError(f"Unknown country filter: {country}")
    if country == "All":
        return df.copy()
    return df[df["country_group"] == country].copy()


def _fit_logistic(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    model.fit(X, y)
    return model


@lru_cache(maxsize=512)
def train_context(
    brand: str,
    campaign: str,
    country: str = "All",
    brand_weight: float = DEFAULT_BRAND_WEIGHT,
    min_followers: int = 0,
    min_engagement: float = 0.0,
    tree_depth: int = 4,
    use_content_history: bool = True,
    use_campaign_history: bool = True,
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

    country_df = country_df[
        (country_df["followers"] >= min_followers) & (country_df["engagement_rate"] >= min_engagement)
    ].copy()
    if len(country_df) < 20:
        raise ValueError("Not enough creators after scenario filters. Lower Minimum Followers or Minimum Engagement.")

    brand_weight = round(_normalize_brand_weight(brand_weight), 2)
    df = enrich_context(country_df, brand, campaign, brand_weight)
    y = df[TARGET_COLUMN].astype(int)
    if y.nunique() < 2:
        raise ValueError("Only one target class remains after filtering. Broaden the scenario filters.")

    features = active_features(bool(use_content_history), bool(use_campaign_history))
    logistic = _fit_logistic(df[features], y)
    baseline_logistic = _fit_logistic(df[BASELINE_FEATURES], y)

    min_leaf = max(8, min(120, int(round(len(df) * 0.08))))
    tree = DecisionTreeClassifier(
        max_depth=tree_depth,
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(df[features], y)

    scored = df.copy()
    scored["selection_probability"] = logistic.predict_proba(df[features])[:, 1]
    scored["baseline_probability"] = baseline_logistic.predict_proba(df[BASELINE_FEATURES])[:, 1]
    scored["probability_delta"] = scored["selection_probability"] - scored["baseline_probability"]
    scored["model_target"] = y

    return ContextModel(
        brand=brand,
        campaign=campaign,
        country=country,
        scored=scored,
        logistic=logistic,
        baseline_logistic=baseline_logistic,
        tree=tree,
        feature_names=features,
        target_positive_rate=float(y.mean()),
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
    use_content_history: bool = True,
    use_campaign_history: bool = True,
) -> tuple[ContextModel, pd.DataFrame]:
    model = train_context(
        brand, campaign, country,
        round(_normalize_brand_weight(brand_weight), 2),
        int(min_followers), round(float(min_engagement), 2), int(tree_depth),
        bool(use_content_history), bool(use_campaign_history),
    )
    scored = model.scored.copy()
    if platform and platform != "All":
        scored = scored[scored["platform"] == platform.strip().lower()].copy()

    if not scored.empty:
        classes = list(model.tree.classes_)
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        scored["tree_probability"] = model.tree.predict_proba(scored[model.feature_names])[:, positive_index]
    else:
        scored["tree_probability"] = pd.Series(dtype=float)

    scored = scored.sort_values(
        ["selection_probability", "brand_fit", "past_campaign_success_rate_sim", "engagement_rate", "followers"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)

    tree_order = scored.sort_values(
        ["tree_probability", "brand_fit", "past_campaign_success_rate_sim", "engagement_rate", "followers"],
        ascending=[False, False, False, False, False],
    ).index.tolist()
    tree_rank_map = {idx: rank for rank, idx in enumerate(tree_order, start=1)}
    scored["tree_rank"] = [tree_rank_map[idx] for idx in scored.index]
    return model, scored


def _compact_count(value: float) -> str:
    n = float(value)
    if n >= 1_000_000:
        return f"{n/1_000_000:.2f}".rstrip("0").rstrip(".") + "M"
    if n >= 1_000:
        return f"{n/1_000:.1f}".rstrip("0").rstrip(".") + "K"
    return f"{n:.0f}"


def _format_threshold(feature: str, threshold: float) -> str:
    if feature == "engagement_rate":
        return f"{threshold:.2f}%"
    if feature == "followers_m":
        return _compact_count(threshold * 1_000_000)
    if feature == "followers_log":
        raw = max(0.0, (10 ** threshold) - 1.0)
        return _compact_count(raw)
    if feature in CONTENT_HISTORY_FEATURES or feature == "past_campaign_success_rate_sim":
        return f"{threshold*100:.1f}%"
    if feature == "past_campaign_count_sim":
        return f"{threshold:.1f}"
    if feature == "verified_flag":
        return f"{threshold:.2f}"
    return f"{threshold:.2f}"


def _format_actual(feature: str, value: float) -> str:
    if feature == "verified_flag":
        return "Yes" if value >= 0.5 else "No"
    return _format_threshold(feature, value)


def export_tree(tree: DecisionTreeClassifier, feature_names: Sequence[str], decision_threshold: float = 0.50) -> dict:
    tree_ = tree.tree_
    def node_dict(node_id: int, depth: int = 0) -> dict:
        left = int(tree_.children_left[node_id])
        right = int(tree_.children_right[node_id])
        raw_counts = tree_.value[node_id][0]
        total = float(np.sum(raw_counts)) or 1.0
        positive_prob = float(raw_counts[1] / total) if len(raw_counts) > 1 else 0.0
        item = {
            "id": int(node_id), "depth": depth, "samples": int(tree_.n_node_samples[node_id]),
            "positive_probability": positive_prob, "is_leaf": left == right,
        }
        if left == right:
            item["label"] = "Suitable" if positive_prob >= decision_threshold else "Not Suitable"
            return item
        feature = feature_names[int(tree_.feature[node_id])]
        threshold = float(tree_.threshold[node_id])
        item.update({
            "feature": feature, "feature_label": FEATURE_LABELS[feature], "threshold": threshold,
            "threshold_label": _format_threshold(feature, threshold),
            "left": node_dict(left, depth + 1), "right": node_dict(right, depth + 1),
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
        "luxury_post_ratio": float(row["luxury_post_ratio_sim"]),
        "jewelry_post_ratio": float(row["jewelry_post_ratio_sim"]),
        "fashion_post_ratio": float(row["fashion_apparel_post_ratio_sim"]),
        "past_campaign_count": int(round(float(row["past_campaign_count_sim"]))),
        "past_campaign_success_rate": float(row["past_campaign_success_rate_sim"]),
        "selection_probability": float(row["selection_probability"]),
        "baseline_probability": float(row.get("baseline_probability", 0.0)),
        "probability_delta": float(row.get("probability_delta", 0.0)),
        "tree_probability": float(row.get("tree_probability", 0.0)),
        "rank": int(row["rank"]),
        "tree_rank": int(row.get("tree_rank", row["rank"])),
        "simulated_target": int(row.get(TARGET_COLUMN, 0)),
    }


def top_records(scored: pd.DataFrame, n: int = 10, method: str = "logistic") -> List[dict]:
    if method == "tree":
        ranked = scored.sort_values(
            ["tree_probability", "brand_fit", "past_campaign_success_rate_sim", "engagement_rate", "followers"],
            ascending=[False, False, False, False, False],
        ).head(n)
    else:
        ranked = scored.sort_values(
            ["selection_probability", "brand_fit", "past_campaign_success_rate_sim", "engagement_rate", "followers"],
            ascending=[False, False, False, False, False],
        ).head(n)
    return [creator_record(row) for _, row in ranked.iterrows()]


def logistic_explanation(model: ContextModel, row: pd.DataFrame) -> dict:
    scaler: StandardScaler = model.logistic.named_steps["scale"]
    classifier: LogisticRegression = model.logistic.named_steps["model"]
    features = model.feature_names
    raw = row[features].iloc[0].astype(float).to_numpy()
    means = scaler.mean_.astype(float)
    scales = scaler.scale_.astype(float)
    standardized = (raw - means) / scales
    coefficients = classifier.coef_[0].astype(float)
    intercept = float(classifier.intercept_[0])
    contributions = coefficients * standardized
    logit = float(intercept + contributions.sum())
    probability = float(1.0 / (1.0 + np.exp(-logit)))

    details = []
    for i, feature in enumerate(features):
        details.append({
            "feature": feature, "feature_label": FEATURE_LABELS[feature],
            "raw_value": float(raw[i]), "raw_label": _format_actual(feature, float(raw[i])),
            "mean": float(means[i]), "scale": float(scales[i]), "standardized": float(standardized[i]),
            "coefficient": float(coefficients[i]), "contribution": float(contributions[i]),
        })
    sklearn_probability = float(model.logistic.predict_proba(row[features])[:, 1][0])
    return {
        "intercept": intercept, "logit": logit, "probability": probability,
        "sklearn_probability": sklearn_probability, "features": details,
        "formula": "P(Y=1)=1/(1+e^(-z))",
        "logit_formula": f"z=β0+ΣβjZj, j=1..{len(features)}",
        "standardization_formula": "Zj=(Xj-μj)/σj",
        "feature_count": len(features),
    }


def explain_creator(model: ContextModel, scored: pd.DataFrame, handle: str, decision_threshold: float = 0.50) -> dict:
    if scored.empty:
        raise ValueError("No creators available for this filter.")
    match = scored[scored["handle"] == handle]
    if match.empty:
        raise ValueError(f"Creator not found in the current result set: {handle}")
    row = match.iloc[[0]]
    values = row[model.feature_names]
    path = model.tree.decision_path(values).indices.tolist()
    leaf_id = int(model.tree.apply(values)[0])
    tree_ = model.tree.tree_
    path_steps = []
    for node_id in path:
        if tree_.children_left[node_id] == tree_.children_right[node_id]:
            continue
        feature = model.feature_names[int(tree_.feature[node_id])]
        threshold = float(tree_.threshold[node_id])
        actual = float(row.iloc[0][feature])
        went_left = actual <= threshold
        path_steps.append({
            "node_id": int(node_id), "feature": feature, "feature_label": FEATURE_LABELS[feature],
            "operator": "≤" if went_left else ">", "threshold": threshold,
            "threshold_label": _format_threshold(feature, threshold), "actual": actual,
            "actual_label": _format_actual(feature, actual),
        })
    selected = creator_record(row.iloc[0])
    classes = list(model.tree.classes_)
    tree_proba = model.tree.predict_proba(values)[0]
    positive_index = classes.index(1) if 1 in classes else len(classes) - 1
    tree_positive_probability = float(tree_proba[positive_index])
    decision_threshold = float(np.clip(float(decision_threshold), 0.0, 1.0))
    selected.update({
        "leaf_id": leaf_id, "decision_path": path_steps, "path_node_ids": [int(x) for x in path],
        "tree_positive_probability": tree_positive_probability,
        "tree_prediction": int(tree_positive_probability >= decision_threshold),
        "decision_threshold": decision_threshold,
    })
    return {
        "selected": selected,
        "tree": export_tree(model.tree, model.feature_names, decision_threshold),
        "logistic_explanation": logistic_explanation(model, row),
    }


def fit_method_summary(brand_weight: float = DEFAULT_BRAND_WEIGHT) -> dict:
    brand_weight = _normalize_brand_weight(brand_weight)
    campaign_weight = 1.0 - brand_weight
    return {
        "brand_weight": brand_weight, "campaign_weight": campaign_weight,
        "label": f"{int(round(brand_weight*100))}% Brand + {int(round(campaign_weight*100))}% Campaign",
        "scale": FIT_SCALE_LABEL,
    }


def feature_set_summary(use_content_history: bool = True, use_campaign_history: bool = True) -> dict:
    features = active_features(use_content_history, use_campaign_history)
    return {
        "feature_count": len(features),
        "features": [{"key": f, "label": FEATURE_LABELS[f]} for f in features],
        "curated_full_count": len(CURATED_FEATURES),
        "model_name": "Enhanced Curated 8" if len(features) == 8 else f"Scenario Model ({len(features)} features)",
        "target": TARGET_COLUMN,
        "target_label": "Simulated Campaign Success (0/1)",
    }


def data_quality_summary() -> dict:
    df = DATA.creators
    return {
        "rows": int(len(df)),
        "repaired_rows": int(DATA.repaired_rows),
        "platform_counts": {k.title(): int(v) for k, v in df["platform"].value_counts().to_dict().items()},
        "country_counts": {k: int(v) for k, v in df["country_group"].value_counts().to_dict().items()},
        "engagement_imputed": int(df["engagement_imputed"].sum()),
        "verified_imputed": int(df["verified_imputed"].sum()),
        "target_positive_rate": float(df[TARGET_COLUMN].mean()),
        "synthetic_columns": [c for c in df.columns if c.endswith("_sim") or c == TARGET_COLUMN],
        "note": "*_sim fields and the campaign-success target are simulated for model prototyping; they are not observed campaign outcomes.",
    }
