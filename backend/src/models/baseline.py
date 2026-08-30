import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}

def get_baseline_models() -> Dict[str, Any]:
    return {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingRegressor(random_state=42)
    }

def train_and_evaluate_baselines(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    models = get_baseline_models()
    results = {}
    best_models = {}
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for name, model in models.items():
        rmse_scores = []
        mae_scores = []
        r2_scores = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = evaluate_metrics(y_test, y_pred)
            
            rmse_scores.append(metrics["RMSE"])
            mae_scores.append(metrics["MAE"])
            r2_scores.append(metrics["R2"])
            
        # Refit on entire dataset
        model.fit(X, y)
        best_models[name] = model
        
        results[name] = {
            "RMSE": float(np.mean(rmse_scores)),
            "MAE": float(np.mean(mae_scores)),
            "R2": float(np.mean(r2_scores))
        }
        
    return best_models, results
