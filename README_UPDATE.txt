BVLGARI Influencer Dashboard - Enhanced 8 Feature Model
=======================================================

1. Replace project ml.py with this ml.py
2. Replace project app.py with this app.py
3. Replace static/app.js with static/app.js (a duplicate app.js is also included at package root for easy download)
4. Copy data/creators_enriched_synthetic_v1.xlsx into the project's data folder
5. Add openpyxl>=3.1,<4 to the existing requirements.txt
6. Redeploy / restart Render

Eight model features
--------------------
1) brand_fit
2) engagement_rate
3) followers_log
4) luxury_post_ratio_sim
5) jewelry_post_ratio_sim
6) fashion_apparel_post_ratio_sim
7) past_campaign_count_sim
8) past_campaign_success_rate_sim

Target
------
simulated_campaign_success_target

Important
---------
The *_sim fields and simulated target are SYNTHETIC prototype data. They must not be described as observed BVLGARI sales/ROAS/conversion history. Replace the target with real historical outcomes when available.

Validation performed
--------------------
- Python syntax: OK
- JavaScript syntax: OK
- Workbook loaded: 5,193 rows
- Repaired rows metadata: 42
- KR training cohort: 195
- KR Instagram candidates: 83
- Logistic explanation: 8 features
