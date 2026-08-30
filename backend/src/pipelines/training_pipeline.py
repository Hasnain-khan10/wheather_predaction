import os
import joblib
import logging
import pandas as pd
from datetime import datetime
from backend.config.settings import settings
from backend.src.models.baseline import train_and_evaluate_baselines
from backend.src.models.deep_models import train_and_evaluate_lstm
from backend.src.explainability.shap_explainer import compute_shap_values

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import hopsworks
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    HOPSWORKS_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception) as e:
    logger.warning(f"Hopsworks/HSML import failed: {e}. Running in local mode.")
    HOPSWORKS_AVAILABLE = False

def fetch_data_from_feature_store() -> tuple[pd.DataFrame, any]:
    if HOPSWORKS_AVAILABLE:
        try:
            project = hopsworks.login(
                host=settings.hopsworks_host,
                api_key_value=settings.hopsworks_api_key,
                project=settings.hopsworks_project_name
            )
            fs = project.get_feature_store()
            fg = fs.get_feature_group(name="aqi_weather_fg", version=1)
            
            query = fg.select_all()
            df = query.read()
            return df, project
        except Exception as e:
            logger.error(f"Hopsworks fetch failed: {e}. Falling back to local feature store.")
            
    if os.path.exists("data/feature_store.parquet"):
        df = pd.read_parquet("data/feature_store.parquet")
        return df, None
    else:
        raise FileNotFoundError("Local feature store not found at data/feature_store.parquet")

def prepare_data(df: pd.DataFrame, target_col: str = "aqi") -> tuple[pd.DataFrame, pd.Series]:
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    cols_to_drop = [target_col, 'timestamp', 'city', 'aqi_category']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]
    
    X = X.fillna(X.mean())
    y = y.fillna(y.mean())
    
    return X, y

def run_training_pipeline():
    logger.info("Fetching data from Feature Store...")
    df, project = fetch_data_from_feature_store()
    
    if len(df) == 0:
        logger.error("No data available in Feature Store.")
        return
        
    logger.info("Preparing data for training...")
    X, y = prepare_data(df)
    
    logger.info("Training baseline models...")
    baseline_models, baseline_metrics = train_and_evaluate_baselines(X, y)
    
    logger.info("Training LSTM model...")
    lstm_model, lstm_metrics = train_and_evaluate_lstm(X, y, time_steps=24)
    
    best_model_name = "LSTM" if lstm_model else "Unknown"
    best_rmse = lstm_metrics["RMSE"]
    best_model = lstm_model
    best_metrics = lstm_metrics
    is_sklearn = False
    
    for name, metrics in baseline_metrics.items():
        logger.info(f"{name} Metrics: {metrics}")
        if metrics["RMSE"] < best_rmse:
            best_rmse = metrics["RMSE"]
            best_model_name = name
            best_model = baseline_models[name]
            best_metrics = metrics
            is_sklearn = True
            
    logger.info(f"Best model selected: {best_model_name} with RMSE: {best_rmse}")
    
    if is_sklearn:
        logger.info("Computing SHAP values...")
        shap_importance = compute_shap_values(best_model, X)
        logger.info(f"Top 5 important features: {list(shap_importance.keys())[:5]}")
    
    model_dir = "model_dir"
    os.makedirs(model_dir, exist_ok=True)
    
    if is_sklearn:
        model_path = os.path.join(model_dir, "model.pkl")
        joblib.dump(best_model, model_path)
    else:
        model_path = os.path.join(model_dir, "model.keras")
        best_model.save(model_path)
        
    logger.info(f"Saved best model locally to {model_path}")
        
    if HOPSWORKS_AVAILABLE and project:
        try:
            input_schema = Schema(X.values)
            output_schema = Schema(y.values)
            model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
            
            mr = project.get_model_registry()
            logger.info("Uploading model to Hopsworks Model Registry...")
            hw_model = mr.python.create_model(
                name="aqi_predictor_model",
                metrics=best_metrics,
                model_schema=model_schema,
                description=f"Best model: {best_model_name} for AQI prediction"
            )
            hw_model.save(model_dir)
            logger.info("Successfully uploaded to Hopsworks Model Registry.")
        except Exception as e:
            logger.error(f"Failed to upload to Hopsworks Model Registry: {e}")
    else:
        logger.info("Hopsworks project unavailable or disabled. Skipping Model Registry upload.")
    
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    run_training_pipeline()
