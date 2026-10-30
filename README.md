# ML Studio — Automated Machine Learning Platform

A full-stack academic project that lets users upload CSV/Excel datasets, inspect data quality, preprocess data, train classification/regression models, and compare evaluation metrics.

## Features
- CSV, XLSX and XLS upload (20 MB limit)
- Dataset preview, column profiling, missing-value and duplicate analysis
- Median/mean numeric imputation and most-frequent categorical imputation
- Duplicate removal and column dropping
- Classification and regression workflows
- Multiple scikit-learn algorithms with a shared preprocessing pipeline
- Model comparison and best-model identification
- Responsive React dashboard and FastAPI REST API

## Requirements
- Python 3.10+
- Node.js 18+

## Run the backend (Terminal 1)
```bash
cd backend
python -m venv venv
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# If activation is blocked, use: venv\Scripts\activate.bat
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```
Backend: http://127.0.0.1:8000  
Swagger API docs: http://127.0.0.1:8000/docs

## Run the frontend (Terminal 2)
```bash
cd frontend
npm install
npm run dev
```
Frontend: http://localhost:5175

## Typical workflow
1. Open Dataset Explorer and upload a CSV or Excel dataset.
2. Inspect the preview, data types, missing values and duplicates.
3. Open Data Preprocessing and apply cleaning.
4. Open Model Training, choose Classification or Regression, select the target and algorithms, and train.
5. Open Model Evaluation to compare the metrics and identify the best model.

## API endpoints
- `GET /` — API welcome
- `GET /api/health` — health status
- `POST /api/dataset/upload` — upload and inspect dataset
- `GET /api/dataset/summary` — current dataset summary
- `POST /api/dataset/preprocess` — clean current dataset
- `POST /api/model/train` — train selected algorithms
- `GET /api/model/results` — retrieve latest experiment results

## Notes / limitations
- This submission version keeps the active dataset and latest experiment in backend process memory. Restarting the backend clears them.
- Training runs synchronously in the API request and is intended for small/medium academic demonstration datasets.
- The test split is random with a fixed seed for repeatable results. Classification uses stratification where possible.
- Metrics are calculated on a held-out test split. A negative R² score is possible and indicates a model performs worse than predicting the target mean.
