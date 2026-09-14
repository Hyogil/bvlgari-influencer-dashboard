# Render deployment

## GitHub + Render

1. Push every file in this folder to a GitHub repository.
2. Sign in to Render and choose **New > Web Service**.
3. Connect the GitHub repository.
4. Render can read `render.yaml`, or use these values manually:
   - Runtime: Python 3
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Deploy the service.
6. Open the generated `https://<service-name>.onrender.com` URL.
7. Health check: `/api/health`

The app serves both the FastAPI backend and the static dashboard, so no separate frontend hosting is required.
