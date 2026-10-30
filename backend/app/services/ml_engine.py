import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, r2_score,
                             mean_absolute_error, mean_squared_error)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

def train_models(df, target, task, selected_models=None, test_size=0.2):
    if target not in df.columns:
        raise ValueError("Selected target column was not found.")
    data = df.dropna(subset=[target]).copy()
    if len(data) < 10:
        raise ValueError("At least 10 rows with a target value are required.")
    X = data.drop(columns=[target])
    y = data[target]
    if X.shape[1] == 0:
        raise ValueError("Dataset must contain at least one feature column.")
    # Remove columns that contain no useful information.
    X = X.loc[:, X.nunique(dropna=False) > 1]
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain.")
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore"))])
    transformers = []
    if numeric: transformers.append(("num", numeric_pipe, numeric))
    if categorical: transformers.append(("cat", categorical_pipe, categorical))
    prep = ColumnTransformer(transformers)
    if task == "classification":
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            y = LabelEncoder().fit_transform(y.astype(str))
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Support Vector Machine": SVC(),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5)
        }
        stratify = y if pd.Series(y).value_counts().min() >= 2 else None
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        X, y = X.loc[valid], y.loc[valid]
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Support Vector Regression": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5)
        }
        stratify = None
    if len(X) < 10:
        raise ValueError("Not enough valid rows after cleaning the target.")
    if selected_models:
        models = {k:v for k,v in models.items() if k in selected_models}
    if not models:
        raise ValueError("Select at least one supported algorithm.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )
    results = []
    for name, estimator in models.items():
        try:
            pipe = Pipeline([("preprocessing", prep), ("model", estimator)])
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            if task == "classification":
                metrics = {
                    "accuracy": round(float(accuracy_score(y_test, pred)), 4),
                    "precision": round(float(precision_score(y_test, pred, average="weighted", zero_division=0)), 4),
                    "recall": round(float(recall_score(y_test, pred, average="weighted", zero_division=0)), 4),
                    "f1_score": round(float(f1_score(y_test, pred, average="weighted", zero_division=0)), 4),
                    "confusion_matrix": confusion_matrix(y_test, pred).tolist()
                }
                score = metrics["accuracy"]
            else:
                mse = mean_squared_error(y_test, pred)
                metrics = {
                    "r2_score": round(float(r2_score(y_test, pred)), 4),
                    "mae": round(float(mean_absolute_error(y_test, pred)), 4),
                    "mse": round(float(mse), 4),
                    "rmse": round(float(np.sqrt(mse)), 4)
                }
                score = metrics["r2_score"]
            results.append({"model": name, "metrics": metrics, "score": score, "status": "success"})
        except Exception as exc:
            results.append({"model": name, "metrics": {}, "score": None, "status": "failed", "error": str(exc)[:180]})
    successful = [r for r in results if r["status"] == "success"]
    if not successful:
        raise ValueError("All selected models failed. Check the target column and dataset.")
    successful.sort(key=lambda r: r["score"] if r["score"] is not None else -999, reverse=True)
    return {"task": task, "target": target, "train_rows": len(X_train), "test_rows": len(X_test),
            "results": results, "best_model": successful[0]["model"]}
