from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
from app.services.data_loader import load_dataset, get_dataset_summary
from app.services.ml_engine import train_models

app = FastAPI(title="ML Studio API", description="Automated machine learning and dataset analysis platform", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://localhost:5174","http://localhost:5175",
 "http://127.0.0.1:5173","http://127.0.0.1:5174","http://127.0.0.1:5175"], allow_credentials=True,
 allow_methods=["*"], allow_headers=["*"])

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
    return {"message":"Welcome to ML Studio API","status":"running"}

@app.get("/api/health")
def health():
    return {"status":"healthy"}

@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    global current_df, current_filename, last_results
    filename = file.filename or ""
    if not filename.lower().endswith((".csv",".xlsx",".xls")):
        raise HTTPException(400, "Upload a CSV or Excel file.")
    try:
        content = await file.read()
        if not content: raise HTTPException(400, "The uploaded file is empty.")
        if len(content) > 20*1024*1024: raise HTTPException(413, "File size must not exceed 20 MB.")
        current_df = load_dataset(filename, content)
        current_filename, last_results = filename, None
        return {"filename":filename,"message":"Dataset uploaded successfully.","dataset":get_dataset_summary(current_df)}
    except HTTPException: raise
    except Exception as exc: raise HTTPException(400, str(exc))
    finally: await file.close()

@app.get("/api/dataset/summary")
def dataset_summary():
    if current_df is None: raise HTTPException(404,"Upload a dataset first.")
    return {"filename":current_filename,"dataset":get_dataset_summary(current_df)}

@app.post("/api/dataset/preprocess")
def preprocess(req: PreprocessRequest):
    global current_df, last_results
    if current_df is None: raise HTTPException(404,"Upload a dataset first.")
    df = current_df.copy()
    unknown = set(req.drop_columns) - set(df.columns)
    if unknown: raise HTTPException(400, f"Unknown columns: {', '.join(unknown)}")
    df = df.drop(columns=req.drop_columns, errors="ignore")
    if req.remove_duplicates: df = df.drop_duplicates()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                value = df[col].median() if req.missing_strategy == "median" else df[col].mean()
                df[col] = df[col].fillna(value)
            else:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    current_df, last_results = df, None
    return {"message":"Preprocessing applied successfully.","dataset":get_dataset_summary(df)}

@app.post("/api/model/train")
def train(req: TrainRequest):
    global last_results
    if current_df is None: raise HTTPException(404,"Upload a dataset first.")
    if req.task not in ("classification","regression"): raise HTTPException(400,"Task must be classification or regression.")
    if not 0.1 <= req.test_size <= 0.4: raise HTTPException(400,"Test size must be between 0.1 and 0.4.")
    try:
        last_results = train_models(current_df, req.target, req.task, req.models or None, req.test_size)
        return last_results
    except ValueError as exc: raise HTTPException(400,str(exc))
    except Exception as exc: raise HTTPException(500,f"Training failed: {str(exc)}")

@app.get("/api/model/results")
def model_results():
    if last_results is None: raise HTTPException(404,"Train models first.")
    return last_results
