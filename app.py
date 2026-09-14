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
    data_quality_summary,
    explain_creator,
    score_creators,
    top_records,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Influencer Selection Dashboard", version="4.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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
    }


@app.get("/api/analyze")
def analyze(
    brand: str = Query("BVLGARI"),
    campaign: str = Query("Luxury / Fashion"),
    platform: str = Query("Instagram"),
    country: str = Query("KR"),
    selected_handle: str | None = Query(None),
):
    if brand not in BRAND_PROFILES:
        raise HTTPException(400, f"Unknown brand: {brand}")
    if campaign not in CAMPAIGN_PROFILES:
        raise HTTPException(400, f"Unknown campaign: {campaign}")
    if country not in VALID_COUNTRIES:
        raise HTTPException(400, f"Unknown country: {country}")

    try:
        model, scored = score_creators(brand, campaign, platform, country)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    if scored.empty:
        raise HTTPException(404, "No creators match the selected platform.")

    if selected_handle is None:
        selected_handle = str(scored.iloc[0]["handle"])

    try:
        explanation = explain_creator(model, scored, selected_handle)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc

    return {
        "context": {"brand": brand, "campaign": campaign, "platform": platform, "country": country},
        "sample_size": int(len(scored)),
        "training_size": int(model.training_size),
        "dataset_size": int(len(DATA.creators)),
        "repaired_rows": int(DATA.repaired_rows),
        "target_type": "proxy",
        "proxy_positive_rate": model.proxy_positive_rate,
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
    }
