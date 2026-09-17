# BVLGARI Influencer Dashboard — Enhanced 8-Feature Model

This project upgrades the original 4-feature pilot to a curated 8-feature model and a simulated campaign-success target for architecture prototyping.

## Curated 8 features
1. `brand_fit`
2. `engagement_rate`
3. `followers_log`
4. `luxury_post_ratio_sim`
5. `jewelry_post_ratio_sim`
6. `fashion_apparel_post_ratio_sim`
7. `past_campaign_count_sim`
8. `past_campaign_success_rate_sim`

Target: `simulated_campaign_success_target`

> Important: all `*_sim` fields and the target are simulated and are not observed BVLGARI campaign outcomes.

## UI changes
- BVLGARI-inspired dark/gold dashboard
- Top 3 enhanced-model recommendations with baseline probability delta
- Two independent Top-10 cards: Logistic Regression and Decision Tree
- Scenario controls for Brand/Campaign balance, minimum followers, minimum engagement, Tree threshold/depth
- Toggles for content-history and campaign-history feature groups
- Dynamic feature chips, Logistic feature contributions, sigmoid curve, selected Decision Tree path, and full tree

## Run locally
```bash
pip install -r requirements.txt
python run.py
```
Then open `http://127.0.0.1:8000`.
