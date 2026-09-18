# Influencer Success Prediction Dashboard

## Run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```
Open `http://127.0.0.1:8000`.

## Data workflow
- `data/train.xlsx`: 5,000 global records; target is visible and used for training.
- `data/test.xlsx`: 193 Korean records; target is masked and used only for prediction.
- `data/validation.xlsx`: same 193 IDs with target revealed; used only after prediction for model evaluation.
- Both Logistic Regression and Decision Tree use the same six features defined in `FEATURES` in `app.py`.

## Influencer profile images
Put JPG files in `static/images/` using the influencer `handle` as the filename. Example: handle `@jin` → `static/images/@jin.jpg`. The UI also tries `jin.jpg` as a fallback.

## Important
The supplied test and validation files contain 193 records (not 195). The dashboard therefore reads and displays the actual row count dynamically.
