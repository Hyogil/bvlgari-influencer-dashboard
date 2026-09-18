BVLGARI Enhanced 8-Feature Dashboard - Complete Render Deploy Package
====================================================================

Included:
- app.py
- ml.py
- static/index.html
- static/style.css
- static/app.js
- data/creators_enriched_synthetic_v1.xlsx
- requirements.txt (includes openpyxl)
- Resource/ folders + sample avatar fallbacks

Render Build Command:
  pip install -r requirements.txt

Render Start Command:
  uvicorn app:app --host 0.0.0.0 --port $PORT

IMPORTANT:
app.py serves STATIC_DIR / "index.html". The correct filename is index.html, not index.htm.
The *_sim fields and campaign target are simulated prototype data.
