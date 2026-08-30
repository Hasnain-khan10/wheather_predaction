import os
import logging
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from backend.config.settings import settings
from backend.src.features.engineer import (
    add_cyclical_time_features,
    add_lag_and_rolling_features,
    calculate_aqi_category
)
from backend.src.pipelines.feature_pipeline import get_or_create_feature_group, save_local_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import hopsworks
    HOPSWORKS_AVAILABLE = True
except (ImportError, ModuleNotFoundError, Exception) as e:
    logger.warning(f"Hopsworks import failed: {e}. Running in local mode.")
    HOPSWORKS_AVAILABLE = False

def fetch_historical_openmeteo(city: str, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m&timezone=UTC"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()
    
    hourly = data['hourly']
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly['time'], utc=True),
        'temperature': hourly['temperature_2m'],
        'humidity': hourly['relative_humidity_2m'],
        'wind_speed': hourly['wind_speed_10m'],
        'city': city
    })
    return df

def fetch_historical_air_quality_openmeteo(city: str, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=us_aqi&timezone=UTC"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()
    
    hourly = data['hourly']
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly['time'], utc=True),
        'aqi': hourly['us_aqi'],
        'city': city
    })
    return df

def run_backfill(city: str = "London", lat: float = 51.5074, lon: float = -0.1278, days: int = 30):
    try:
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=days)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching historical data for {city} from {start_str} to {end_str}")
        weather_df = fetch_historical_openmeteo(city, lat, lon, start_str, end_str)
        aqi_df = fetch_historical_air_quality_openmeteo(city, lat, lon, start_str, end_str)
        
        df = pd.merge(weather_df, aqi_df, on=['timestamp', 'city'], how='inner')
        df = df.dropna()
        
        logger.info("Applying feature engineering pipeline")
        df = add_cyclical_time_features(df)
        df = add_lag_and_rolling_features(df, target_col="aqi")
        df = calculate_aqi_category(df)
        
        df = df.dropna()
        
        logger.info(f"Processed dataframe shape: {df.shape}")
        
        if HOPSWORKS_AVAILABLE:
            try:
                project = hopsworks.login(
                    host=settings.hopsworks_host,
                    api_key_value=settings.hopsworks_api_key,
                    project=settings.hopsworks_project_name
                )
                fg = get_or_create_feature_group(project)
                fg.insert(df, write_options={"wait_for_job": True})
                logger.info("Successfully completed backfill insertion.")
            except Exception as e:
                logger.error(f"Hopsworks connection failed: {e}. Falling back to local storage.")
                save_local_features(df)
        else:
            logger.info("Hopsworks is not available. Saving features locally.")
            save_local_features(df)
            
    except Exception as e:
        logger.error(f"Error during backfill: {e}")
        raise

if __name__ == "__main__":
    run_backfill()
