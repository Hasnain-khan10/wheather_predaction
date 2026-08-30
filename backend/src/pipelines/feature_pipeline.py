import os
import logging
import requests
import pandas as pd
from datetime import datetime, timezone
from backend.config.settings import settings
from backend.src.features.engineer import add_cyclical_time_features, calculate_aqi_category

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import hopsworks
    HOPSWORKS_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception) as e:
    logger.warning(f"Hopsworks import failed: {e}. Running in local mode.")
    HOPSWORKS_AVAILABLE = False

def fetch_aqicn_data(city: str, api_key: str) -> dict:
    url = f"https://api.waqi.info/feed/{city}/?token={api_key}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    if data.get('status') != 'ok':
        raise ValueError(f"AQICN API error: {data.get('data')}")
    return data['data']

def fetch_openweather_data(city: str, api_key: str) -> dict:
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

def ingest_live_data(city: str = "London") -> pd.DataFrame:
    logger.info(f"Fetching live data for {city}")
    aqi_data = fetch_aqicn_data(city, settings.aqicn_api_key)
    weather_data = fetch_openweather_data(city, settings.openweather_api_key)
    
    timestamp = pd.to_datetime("now", utc=True).floor('H')
    
    record = {
        'city': city,
        'timestamp': timestamp,
        'aqi': float(aqi_data.get('aqi', 0)),
        'temperature': float(weather_data.get('main', {}).get('temp', 0.0)),
        'humidity': float(weather_data.get('main', {}).get('humidity', 0.0)),
        'wind_speed': float(weather_data.get('wind', {}).get('speed', 0.0)),
    }
    
    df = pd.DataFrame([record])
    df = add_cyclical_time_features(df)
    df = calculate_aqi_category(df)
    return df

def get_or_create_feature_group(project):
    fs = project.get_feature_store()
    try:
        fg = fs.get_feature_group(name="aqi_weather_fg", version=1)
    except Exception:
        logger.info("Creating Feature Group aqi_weather_fg")
        fg = fs.create_feature_group(
            name="aqi_weather_fg",
            version=1,
            description="Air quality and weather features",
            primary_key=["city", "timestamp"],
            event_time="timestamp",
            online_enabled=True
        )
    return fg

def save_local_features(df: pd.DataFrame, filepath: str = "data/feature_store.parquet"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        try:
            existing_df = pd.read_parquet(filepath)
            df = pd.concat([existing_df, df]).drop_duplicates(subset=['city', 'timestamp'], keep='last')
        except Exception as e:
            logger.warning(f"Could not read existing local feature store: {e}")
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved features locally to {filepath}")

def run():
    try:
        df = ingest_live_data("London")
        logger.info(f"Ingested live data shape: {df.shape}")
        
        if HOPSWORKS_AVAILABLE:
            try:
                project = hopsworks.login(
                    host=settings.hopsworks_host,
                    api_key_value=settings.hopsworks_api_key,
                    project=settings.hopsworks_project_name
                )
                fg = get_or_create_feature_group(project)
                fg.insert(df, write_options={"wait_for_job": False})
                logger.info("Successfully inserted live data into Feature Store.")
            except Exception as e:
                logger.error(f"Hopsworks connection failed: {e}. Falling back to local storage.")
                save_local_features(df)
        else:
            logger.info("Hopsworks is not available. Saving features locally.")
            save_local_features(df)
            
    except Exception as e:
        logger.error(f"Error in feature pipeline: {e}")
        raise

if __name__ == "__main__":
    run()
