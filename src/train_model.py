"""Model training utilities."""

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from .preprocessing import get_feature_types, build_preprocessor
from .evaluate_model import regression_metrics, compare_regression_models


def get_regression_models(random_state=51):
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=random_state, max_depth=5
        ),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            random_state=random_state
        ),
    }


def train_models(X_train, X_test, y_train, y_test, random_state=51):
    numeric_features, categorical_features = get_feature_types(X_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    trained = {}
    results = []

    for name, model in get_regression_models(random_state).items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        metrics = regression_metrics(y_test, pred)
        results.append({"Model": name, **metrics})
        trained[name] = pipeline

    results_df = compare_regression_models(results)
    best_name = results_df.iloc[0]["Model"]
    return trained, results_df, best_name
