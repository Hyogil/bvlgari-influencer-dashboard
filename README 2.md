# BVLGARI Influencer Selection Dashboard — Formula + Compact Tree

This version uses the merged 5,300-creator dataset and supports:

- Country filter: **KR / Global / All**
- Platform filter: Instagram / TikTok / YouTube / All
- Logistic Regression ranking (Top 3 and clickable Top 10)
- Exact Logistic Regression arithmetic for the selected creator
  - raw feature values
  - z-scores from `StandardScaler`
  - fitted beta coefficients
  - per-feature contribution to the logit
  - total logit `z`
  - sigmoid probability
  - interactive sigmoid point for the selected creator
- Full Decision Tree with only the selected creator's path highlighted
- More compact Decision Tree layout with no desktop internal scroll bar

## Windows

1. Extract the ZIP, for example to `C:\BVLGARI`.
2. Double-click `run_windows.bat`, or run:

```bat
cd /d C:\BVLGARI
run_windows.bat
```

3. Open `http://127.0.0.1:8000` if the browser does not open automatically.

`run_windows.bat` is ASCII/CRLF with no UTF-8 BOM to avoid the Korean Windows CMD encoding problem seen in earlier versions.

## Logistic Regression equation shown in the UI

The fitted pipeline is:

`StandardScaler -> LogisticRegression`

For each feature:

`Zj = (Xj - mean_j) / scale_j`

Then:

`z = beta0 + beta1*Z1 + beta2*Z2 + beta3*Z3 + beta4*Z4`

and:

`P(Y=1) = 1 / (1 + exp(-z))`

The numbers shown in the formula panel come directly from the fitted scikit-learn model and are cross-checked against `predict_proba()`.

## Academic limitation

The public creator dataset does not contain a historical campaign outcome. The project therefore creates a transparent proxy binary target. The displayed probabilities should be described as **model-estimated campaign suitability under the proxy-target definition**, not as actual historical BVLGARI campaign success probabilities.


## Local Resource folder

This build uses `Resource/` for creator avatars and local icon resources.

- `Resource/images/instagram/` — Instagram JPG avatars, filename = handle without `@`
- `Resource/images/tiktok/` — TikTok JPG avatars
- `Resource/images/youtube/` — YouTube JPG avatars
- `Resource/images/sample/` — local fallback avatars, only when a creator image is missing
- `Resource/fontawesome/css/fa-local.css` — local FA-class-compatible icon stylesheet

Example: `@jin` -> `Resource/images/instagram/jin.jpg`.

External `i.pravatar.cc`, Font Awesome CDN, and Google Fonts dependencies were removed.
