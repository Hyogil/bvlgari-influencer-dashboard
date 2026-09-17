from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ml import (
    BRAND_PROFILES,
    CAMPAIGN_PROFILES,
    DATA,
    VALID_COUNTRIES,
    TARGET_COLUMN,
    data_quality_summary,
    explain_creator,
    feature_set_summary,
    fit_method_summary,
    score_creators,
    top_records,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RESOURCE_DIR = BASE_DIR / "Resource"

app = FastAPI(title="BVLGARI Enhanced Influencer Dashboard", version="6.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/Resource", StaticFiles(directory=RESOURCE_DIR), name="resource")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/options")
def options():
    return {
        "brands": list(BRAND_PROFILES.keys()),
        "campaigns": list(CAMPAIGN_PROFILES.keys()),
        "platforms": ["All", "Instagram", "TikTok", "YouTube"],
        "countries": ["KR", "Global", "All"],
        "sample_size": int(len(DATA.creators)),
        "repaired_rows": int(DATA.repaired_rows),
        "fit_method": fit_method_summary(),
        "feature_set": feature_set_summary(True, True),
        "target": TARGET_COLUMN,
    }


@app.get("/api/analyze")
def analyze(
    brand: str = Query("BVLGARI"),
    campaign: str = Query("Luxury / Fashion"),
    platform: str = Query("Instagram"),
    country: str = Query("KR"),
    selected_handle: str | None = Query(None),
    brand_weight: float = Query(0.50, ge=0.0, le=1.0),
    min_followers: int = Query(0, ge=0),
    min_engagement: float = Query(0.0, ge=0.0, le=50.0),
    decision_threshold: float = Query(0.50, ge=0.30, le=0.80),
    tree_depth: int = Query(4, ge=2, le=6),
    use_content_history: bool = Query(True),
    use_campaign_history: bool = Query(True),
):
    if brand not in BRAND_PROFILES:
        raise HTTPException(400, f"Unknown brand: {brand}")
    if campaign not in CAMPAIGN_PROFILES:
        raise HTTPException(400, f"Unknown campaign: {campaign}")
    if country not in VALID_COUNTRIES:
        raise HTTPException(400, f"Unknown country: {country}")

    try:
        model, scored = score_creators(
            brand, campaign, platform, country, brand_weight, min_followers,
            min_engagement, tree_depth, use_content_history, use_campaign_history,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if scored.empty:
        raise HTTPException(404, "No creators match the selected platform.")

    if selected_handle is None:
        selected_handle = str(scored.iloc[0]["handle"])
    try:
        explanation = explain_creator(model, scored, selected_handle, decision_threshold)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    return {
        "context": {
            "brand": brand, "campaign": campaign, "platform": platform, "country": country,
            "brand_weight": brand_weight, "campaign_weight": 1.0 - brand_weight,
            "min_followers": min_followers, "min_engagement": min_engagement,
            "decision_threshold": decision_threshold, "tree_depth": tree_depth,
            "use_content_history": use_content_history,
            "use_campaign_history": use_campaign_history,
        },
        "sample_size": int(len(scored)),
        "training_size": int(model.training_size),
        "dataset_size": int(len(DATA.creators)),
        "repaired_rows": int(DATA.repaired_rows),
        "fit_method": fit_method_summary(brand_weight),
        "feature_set": feature_set_summary(use_content_history, use_campaign_history),
        "target_type": "synthetic_historical_outcome",
        "target_column": TARGET_COLUMN,
        "target_positive_rate": model.target_positive_rate,
        "top10": top_records(scored, 10, "logistic"),
        "top10_logistic": top_records(scored, 10, "logistic"),
        "top10_tree": top_records(scored, 10, "tree"),
        **explanation,
    }


@app.get("/api/data-quality")
def data_quality():
    return data_quality_summary()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "version": "6.0.0",
        "creators": int(len(DATA.creators)),
        "repaired_rows": int(DATA.repaired_rows),
        "target": TARGET_COLUMN,
        "feature_set": feature_set_summary(True, True),
    }
