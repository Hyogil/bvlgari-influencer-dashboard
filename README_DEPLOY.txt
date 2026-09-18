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

8 DISPLAYED CONTROLS (7 sliders + 1 automatic value)
1. Brand <-> Campaign Weight (changes Profile Fit and therefore the model input)
2. Profile Fit Weight (changes Scenario Fit and Final Ranking Score)
3. Content Fit Weight (changes Scenario Fit and Final Ranking Score)
4. Campaign History Weight (automatic: 100% - Profile - Content)
5. Minimum Followers (candidate filter after training)
6. Minimum Engagement (candidate filter after training)
7. Suitable Threshold (Decision Tree leaf label threshold)
8. Tree Depth (Decision Tree max_depth)

IMPORTANT DISTINCTION
- Logistic Regression produces Model Probability from the five model features.
- Scenario Fit = Profile/Content/History weighted business sensitivity score.
- Final Ranking Score = 50% Model Probability + 50% Scenario Fit.
- Scenario weights do NOT overwrite Logistic Regression beta coefficients; they affect ranking through the separate Scenario Fit term.
- Minimum Followers / Engagement filter the candidate list after the country cohort is trained.
- Tree Depth affects only Decision Tree complexity.
- Suitable Threshold affects Suitable / Not Suitable labeling, not Final Ranking Score.

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

UI UPDATE - COMPACT 8 CONTROL STRIP
- Desktop/presentation width >=1450px: all 8 control cards are displayed in one horizontal row.
- 821-1449px: responsive 4 x 2 layout.
- <=820px: responsive 2-column mobile layout.
- Controls preserved: Brand/Campaign, Profile Fit, Content Fit, Campaign History (auto), Min Followers, Min Engagement, Threshold, Tree Depth.
- v6.1 additionally changes ranking logic so Scenario Fit directly contributes to Final Ranking Score.


Ranking update (v6.1)
---------------------
Final Ranking Score = 50% Logistic Regression probability + 50% Scenario Fit.
Profile / Content / History controls now directly affect ranking through Scenario Fit.
Minimum Followers / Engagement continue to filter the candidate set.
Suitable Threshold and Tree Depth remain Decision Tree interpretation controls and do not directly enter Final Ranking Score.
