BVLGARI 5-Feature Dashboard - Render Deploy Package
====================================================

MODEL FEATURES (5)
1. Profile Fit
   - Profile Fit = Brand Weight * Brand Score + Campaign Weight * Campaign Score
   - Brand Weight is adjustable with Slider #1; Campaign Weight is the remainder.

2. Content Fit
   - Content Fit = 50% Luxury Post Ratio + 30% Jewelry Post Ratio + 20% Fashion/Apparel Post Ratio

3. Campaign History Score
   - Experience Score = min(Past Campaign Count / 10, 1)
   - Campaign History Score = 30% Experience Score + 70% Past Campaign Success Rate

4. Engagement Rate
5. Follower Reach (log1p followers)

7 CONTROLS
1. Brand <-> Campaign Weight (changes Profile Fit and therefore the model input)
2. Profile Fit Weight (Scenario Fit only)
3. Content Fit Weight (Scenario Fit only)
   - History Weight = 100% - Profile Weight - Content Weight (automatic)
4. Minimum Followers (candidate filter after training)
5. Minimum Engagement (candidate filter after training)
6. Suitable Threshold (Decision Tree leaf label threshold)
7. Tree Depth (Decision Tree max_depth)

IMPORTANT DISTINCTION
- Logistic Regression remains the primary ranking model.
- Scenario Fit = Profile/Content/History weighted business sensitivity score.
- Scenario weights do NOT overwrite Logistic Regression beta coefficients.
- Minimum Followers / Engagement filter the candidate list after the country cohort is trained.
- Tree Depth affects only Decision Tree complexity.
- Suitable Threshold affects Suitable / Not Suitable labeling, not Logistic probability.

DEFAULT SCENARIO
- Brand / Campaign = 50 / 50
- Profile / Content / History = 40 / 35 / 25
- Minimum Followers = 0
- Minimum Engagement = 0%
- Suitable Threshold = 50%
- Tree Depth = 4

DATA
- data/creators_enriched_synthetic_v1.xlsx
- 5,193 rows
- 42 repaired-row metadata flags
- *_sim columns and simulated_campaign_success_target are synthetic prototype data.

RENDER
Build Command:
  pip install -r requirements.txt

Start Command:
  uvicorn app:app --host 0.0.0.0 --port $PORT

After replacing the files in GitHub, use Render -> Manual Deploy -> Clear build cache & deploy.
