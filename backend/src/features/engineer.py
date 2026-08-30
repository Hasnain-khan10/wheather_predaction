import pandas as pd
import numpy as np

def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes sin/cos encodings for hour, day_of_week, and month."""
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)
    
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    return df

def add_lag_and_rolling_features(df: pd.DataFrame, target_col: str = "aqi") -> pd.DataFrame:
    """Computes lag features, rolling means, standard deviations, and AQI rate-of-change."""
    df = df.copy()
    
    if 'city' in df.columns:
        df = df.sort_values(by=['city', 'timestamp'])
        group = df.groupby('city')
    else:
        df = df.sort_values(by=['timestamp'])
        group = df
        
    for lag in [1, 3, 6, 24]:
        if 'city' in df.columns:
            df[f'{target_col}_lag_{lag}'] = group[target_col].shift(lag)
        else:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
            
    if 'city' in df.columns:
        df[f'{target_col}_rolling_mean_24'] = group[target_col].rolling(window=24, min_periods=1).mean().reset_index(0, drop=True)
        df[f'{target_col}_rolling_std_24'] = group[target_col].rolling(window=24, min_periods=1).std().reset_index(0, drop=True)
    else:
        df[f'{target_col}_rolling_mean_24'] = df[target_col].rolling(window=24, min_periods=1).mean()
        df[f'{target_col}_rolling_std_24'] = df[target_col].rolling(window=24, min_periods=1).std()
        
    if f'{target_col}_lag_1' in df.columns:
        df[f'{target_col}_roc_1'] = df[target_col] - df[f'{target_col}_lag_1']
        
    return df

def calculate_aqi_category(df: pd.DataFrame, aqi_col: str = "aqi") -> pd.DataFrame:
    """Maps continuous AQI values to EPA standard categories."""
    df = df.copy()
    
    def get_category(aqi: float) -> str:
        if pd.isna(aqi):
            return "Unknown"
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Moderate"
        elif aqi <= 150:
            return "Unhealthy for Sensitive Groups"
        elif aqi <= 200:
            return "Unhealthy"
        elif aqi <= 300:
            return "Very Unhealthy"
        else:
            return "Hazardous"
            
    df['aqi_category'] = df[aqi_col].apply(get_category)
    return df
