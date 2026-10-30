from io import BytesIO
import pandas as pd

def load_dataset(filename: str, content: bytes, sheet_name=0):
    name = filename.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(BytesIO(content))
        except UnicodeDecodeError:
            return pd.read_csv(BytesIO(content), encoding="latin-1")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(content), sheet_name=sheet_name)
    raise ValueError("Unsupported file format. Upload CSV or Excel.")

def get_dataset_summary(df):
    if df is None or df.empty:
        raise ValueError("The dataset has no rows.")
    preview = df.head(10).where(pd.notna(df), None).to_dict(orient="records")
    columns = []
    for col in df.columns:
        columns.append({
            "name": str(col),
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "unique": int(df[col].nunique(dropna=True))
        })
    return {
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "columns": columns,
        "preview": preview,
        "missing_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_kb": round(float(df.memory_usage(deep=True).sum()) / 1024, 2)
    }
