"""Prediction and Joblib save/load utilities."""

import joblib
import pandas as pd


def build_model_package(pipeline, feature_columns, target_column, task_type, model_name, X_reference):
    schema = {}
    for column in feature_columns:
        series = X_reference[column]
        if pd.api.types.is_numeric_dtype(series):
            non_null = pd.to_numeric(series, errors="coerce").dropna()
            schema[column] = {
                "type": "numeric",
                "min": float(non_null.min()) if len(non_null) else 0.0,
                "max": float(non_null.max()) if len(non_null) else 100.0,
                "median": float(non_null.median()) if len(non_null) else 0.0,
            }
        else:
            values = series.dropna().astype(str).unique().tolist()
            schema[column] = {
                "type": "categorical",
                "categories": values,
            }

    return {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "task_type": task_type,
        "model_name": model_name,
        "feature_schema": schema,
    }


def save_model_package(package, path):
    joblib.dump(package, path)


def load_model_package(path):
    return joblib.load(path)


def predict_student(package, input_data):
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        raise TypeError("input_data must be a dictionary or pandas DataFrame.")

    features = package["feature_columns"]
    missing = [c for c in features if c not in input_df.columns]
    if missing:
        raise ValueError(f"Missing required input features: {missing}")

    input_df = input_df[features]
    return package["pipeline"].predict(input_df)[0]
