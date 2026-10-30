import os

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.data_loader import get_dataset_summary, load_dataset
from app.services.ml_engine import train_models


app = FastAPI(
    title="ML Studio API",
    description="Automated machine learning and dataset analysis platform",
    version="1.0.0",
)

# Keep local development origins and add the deployed frontend URL from
# Render's FRONTEND_URL environment variable.
allowed_origins = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:5175",
]

# Supports one frontend URL or a comma-separated list of frontend URLs.
frontend_urls = os.getenv("FRONTEND_URLS", "")
single_frontend_url = os.getenv("FRONTEND_URL", "")
configured_urls = [single_frontend_url, *frontend_urls.split(",")]

for url in configured_urls:
    url = url.strip().rstrip("/")
    if url and url not in allowed_origins:
        allowed_origins.append(url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# The active dataset and latest experiment are held in memory for this
# single-process academic demo. Restarting the service clears this state.
current_df = None
current_filename = None
last_results = None


class PreprocessRequest(BaseModel):
    missing_strategy: str = "median"
    remove_duplicates: bool = True
    drop_columns: list[str] = Field(default_factory=list)


class TrainRequest(BaseModel):
    target: str
    task: str
    models: list[str] = Field(default_factory=list)
    test_size: float = 0.2


@app.get("/")
def home():
    return {
        "message": "Welcome to ML Studio API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "ML Studio API",
    }


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    global current_df, current_filename, last_results

    filename = file.filename or ""
    if not filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(
            status_code=400,
            detail="Upload a CSV or Excel file.",
        )

    try:
        content = await file.read()

        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="File size must not exceed 20 MB.",
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty.",
            )

        df = load_dataset(filename, content)
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded dataset contains no rows.",
            )

        current_df = df
        current_filename = filename
        last_results = None

        return {
            "filename": filename,
            "message": "Dataset uploaded and analysed successfully.",
            "dataset": get_dataset_summary(df),
        }

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to read this dataset. Check that the file is valid "
                "and correctly formatted."
            ),
        ) from exc
    finally:
        await file.close()


@app.get("/api/dataset/summary")
def dataset_summary():
    if current_df is None:
        raise HTTPException(
            status_code=404,
            detail="Upload a dataset first.",
        )

    return {
        "filename": current_filename,
        "dataset": get_dataset_summary(current_df),
    }


@app.post("/api/dataset/preprocess")
def preprocess_dataset(request: PreprocessRequest):
    global current_df, last_results

    if current_df is None:
        raise HTTPException(
            status_code=404,
            detail="Upload a dataset first.",
        )

    if request.missing_strategy not in ("median", "mean"):
        raise HTTPException(
            status_code=400,
            detail="Missing strategy must be 'median' or 'mean'.",
        )

    unknown_columns = set(request.drop_columns) - set(current_df.columns)
    if unknown_columns:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown columns: {', '.join(sorted(unknown_columns))}",
        )

    df = current_df.copy()
    df = df.drop(columns=request.drop_columns, errors="ignore")

    if df.empty or len(df.columns) == 0:
        raise HTTPException(
            status_code=400,
            detail="Preprocessing cannot remove every column.",
        )

    if request.remove_duplicates:
        df = df.drop_duplicates()

    for column in df.columns:
        if not df[column].isna().any():
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            fill_value = (
                df[column].median()
                if request.missing_strategy == "median"
                else df[column].mean()
            )
            if pd.notna(fill_value):
                df[column] = df[column].fillna(fill_value)
        else:
            mode_values = df[column].mode()
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            df[column] = df[column].fillna(fill_value)

    current_df = df
    last_results = None

    return {
        "message": "Preprocessing applied successfully.",
        "dataset": get_dataset_summary(df),
    }


@app.post("/api/model/train")
def train_model(request: TrainRequest):
    global last_results

    if current_df is None:
        raise HTTPException(
            status_code=404,
            detail="Upload a dataset first.",
        )

    if request.task not in ("classification", "regression"):
        raise HTTPException(
            status_code=400,
            detail="Task must be classification or regression.",
        )

    if not 0.1 <= request.test_size <= 0.4:
        raise HTTPException(
            status_code=400,
            detail="Test size must be between 0.1 and 0.4.",
        )

    try:
        last_results = train_models(
            current_df,
            request.target,
            request.task,
            request.models or None,
            request.test_size,
        )
        return last_results
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(exc)}",
        ) from exc


@app.get("/api/model/results")
def get_model_results():
    if last_results is None:
        raise HTTPException(
            status_code=404,
            detail="Train models first.",
        )

    return last_results