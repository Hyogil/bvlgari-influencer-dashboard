from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ml import (
    BRAND_PROFILES,
    CAMPAIGN_PROFILES,
    DATA,
    DEFAULT_CONTROLS,
    VALID_COUNTRIES,
    data_quality_summary,
    explain_creator,
    fit_method_summary,
    ranking_summary,
    score_creators,
    top_records,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RESOURCE_DIR = BASE_DIR / "Resource"

app = FastAPI(title="Influencer Selection Dashboard", version="7.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/Resource", StaticFiles(directory=RESOURCE_DIR), name="resource")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/options")
def options():
    platform_order = ["All", "Instagram", "TikTok", "YouTube"]
    return {
        "brands": list(BRAND_PROFILES.keys()),
        "campaigns": list(CAMPAIGN_PROFILES.keys()),
        "platforms": platform_order,
        "countries": ["KR", "Global", "All"],
        "sample_size": int(len(DATA.creators)),
        "repaired_rows": int(DATA.repaired_rows),
        "defaults": DEFAULT_CONTROLS,
        "fit_method": fit_method_summary(),
        "ranking": ranking_summary(),
    }


@app.get("/api/analyze")
def analyze(
    brand: str = Query("BVLGARI"),
    campaign: str = Query("Luxury / Fashion"),
    platform: str = Query("Instagram"),
    country: str = Query("KR"),
    selected_handle: str | None = Query(None),
    min_followers: float = Query(0.0, ge=0.0),
    min_engagement: float = Query(0.0, ge=0.0, le=50.0),
    suitable_threshold: float = Query(0.50, ge=0.0, le=1.0),
    tree_depth: int = Query(4, ge=1, le=8),
):
    if brand not in BRAND_PROFILES:
        raise HTTPException(400, f"Unknown brand: {brand}")
    if campaign not in CAMPAIGN_PROFILES:
        raise HTTPException(400, f"Unknown campaign: {campaign}")
    if country not in VALID_COUNTRIES:
        raise HTTPException(400, f"Unknown country: {country}")
    try:
        model, scored, controls = score_creators(
            brand=brand,
            campaign=campaign,
            platform=platform,
            country=country,
            min_followers=min_followers,
            min_engagement=min_engagement,
            suitable_threshold=suitable_threshold,
            tree_depth=tree_depth,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    if scored.empty:
        raise HTTPException(404, "No creators match the current platform/follower/engagement filters.")

    if selected_handle is None or selected_handle not in set(scored["handle"].astype(str)):
        selected_handle = str(scored.iloc[0]["handle"])

    try:
        explanation = explain_creator(
            model,
            scored,
            selected_handle,
            suitable_threshold=controls["suitable_threshold"],
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    return {
        "context": {
            "brand": brand,
            "campaign": campaign,
            "platform": platform,
            "country": country,
            **controls,
        },
        "sample_size": int(len(scored)),
        "training_size": int(model.training_size),
        "dataset_size": int(len(DATA.creators)),
        "repaired_rows": int(DATA.repaired_rows),
        "fit_method": fit_method_summary(),
        "ranking": ranking_summary(),
        "target_type": "simulated_historical_outcome",
        "target_positive_rate": model.target_positive_rate,
        "proxy_positive_rate": model.target_positive_rate,
        "dataset_file": "creators_enriched_synthetic_v1.xlsx",
        "top10": top_records(scored, 10),
        **explanation,
    }


@app.get("/api/data-quality")
def data_quality():
    return data_quality_summary()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "creators": int(len(DATA.creators)),
        "repaired_rows": int(DATA.repaired_rows),
        "model_features": 9,
        "dataset_file": "creators_enriched_synthetic_v1.xlsx",
    }
