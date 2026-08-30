import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        BatchNormalization(),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def prepare_sequences(X: np.ndarray, y: np.ndarray, time_steps: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_and_evaluate_lstm(X: pd.DataFrame, y: pd.Series, time_steps: int = 24) -> Tuple[Sequential, Dict[str, float]]:
    X_val = X.values
    y_val = y.values
    
    X_seq, y_seq = prepare_sequences(X_val, y_val, time_steps)
    
    # Handle edge case where dataset is too small
    if len(X_seq) < 10:
        return None, {"RMSE": float('inf'), "MAE": float('inf'), "R2": -float('inf')}
    
    split = int(0.8 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    
    model = create_lstm_model((X_seq.shape[1], X_seq.shape[2]))
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    y_pred = model.predict(X_test, verbose=0).flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2)
    }
    
    return model, results
