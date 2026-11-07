"""Compare and select regression models for OpenAP fuel prediction."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


SEED = 4607
DATASET = Path("data/generated/openap_b77w_cmb_dxb_5000.csv")
OUTPUT_DIR = Path("results/model_selection")
FEATURES = [
    "zero_fuel_mass_kg",
    "cruise_speed_kt",
    "cruise_altitude_ft",
    "temperature_deviation_k",
    "flight_duration_min",
]
TARGET = "fuel_consumption_kg"


def median_prediction_time_ms(model: Pipeline, X: pd.DataFrame) -> float:
    """Benchmark batch prediction after warm-up using a stable median."""
    for _ in range(5):
        model.predict(X)
    timings = []
    for _ in range(50):
        started = time.perf_counter()
        model.predict(X)
        timings.append((time.perf_counter() - started) * 1000)
    return float(np.median(timings))


def model_searches() -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    """Return the five candidate models and compact tuning grids."""
    return {
        "Multiple Linear Regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LinearRegression()),
                ]
            ),
            {},
        ),
        "Polynomial Regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                    ("scaler", StandardScaler()),
                    ("model", Ridge()),
                ]
            ),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "Random Forest Regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(random_state=SEED, n_jobs=-1),
                    ),
                ]
            ),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 12, 20],
                "model__min_samples_leaf": [1, 2, 4],
                "model__max_features": [1.0, "sqrt"],
            },
        ),
        "Gradient Boosting Regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", GradientBoostingRegressor(random_state=SEED)),
                ]
            ),
            {
                "model__n_estimators": [100, 200, 300],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [2, 3],
                "model__min_samples_leaf": [2, 5],
            },
        ),
        "Support Vector Regression": (
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf")),
                ]
            ),
            {
                "model__C": [10.0, 100.0, 1000.0],
                "model__gamma": ["scale", 0.01, 0.1],
                "model__epsilon": [10.0, 50.0, 100.0],
            },
        ),
    }


def main() -> None:
    frame = pd.read_csv(DATASET)
    if frame[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Model-selection columns contain missing values")

    X = frame[FEATURES]
    y = frame[TARGET]
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED
    )

    cross_validation = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, (pipeline, parameter_grid) in model_searches().items():
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring=scoring,
            refit="rmse",
            cv=cross_validation,
            n_jobs=-1,
            return_train_score=False,
        )

        started = time.perf_counter()
        search.fit(X_dev, y_dev)
        search_seconds = time.perf_counter() - started

        predictions = search.best_estimator_.predict(X_test)
        prediction_ms = median_prediction_time_ms(search.best_estimator_, X_test)

        best_index = search.best_index_
        record = {
            "model": name,
            "cv_rmse_kg": -float(search.cv_results_["mean_test_rmse"][best_index]),
            "cv_mae_kg": -float(search.cv_results_["mean_test_mae"][best_index]),
            "cv_r2": float(search.cv_results_["mean_test_r2"][best_index]),
            "test_rmse_kg": float(root_mean_squared_error(y_test, predictions)),
            "test_mae_kg": float(mean_absolute_error(y_test, predictions)),
            "test_r2": float(r2_score(y_test, predictions)),
            "search_time_seconds": search_seconds,
            "test_prediction_time_ms": prediction_ms,
            "prediction_time_ms_per_row": prediction_ms / len(X_test),
            "best_parameters": json.dumps(search.best_params_, sort_keys=True),
        }
        results.append(record)
        fitted_models[name] = search.best_estimator_
        print(json.dumps(record, indent=2))

    comparison = pd.DataFrame(results).sort_values("cv_rmse_kg").reset_index(drop=True)
    comparison.insert(0, "rank", comparison.index + 1)
    selected_row = comparison.iloc[0]
    selected_name = str(selected_row["model"])
    selected_model = fitted_models[selected_name]

    comparison.to_csv(OUTPUT_DIR / "regression_model_comparison.csv", index=False)
    joblib.dump(selected_model, OUTPUT_DIR / "selected_fuel_model.joblib")

    summary = {
        "selection_rule": (
            "Lowest five-fold cross-validated RMSE on the 85% development set; "
            "the held-out test set is used only for final confirmation"
        ),
        "random_seed": SEED,
        "dataset_rows": int(len(frame)),
        "development_rows": int(len(X_dev)),
        "held_out_test_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "selected_model": selected_name,
        "selected_metrics": selected_row.to_dict(),
    }
    (OUTPUT_DIR / "model_selection_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\nFinal ranking")
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {selected_name}")


if __name__ == "__main__":
    main()
