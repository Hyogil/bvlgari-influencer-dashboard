# Render deployment

- Runtime: Python
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

The application expects `data/creators.csv`, which is the enriched 5,193-row synthetic prototype dataset included in this package.

Academic caution: `*_sim` fields and `simulated_campaign_success_target` are simulated for model prototyping only.
