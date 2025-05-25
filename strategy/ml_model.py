# strategy/ml_model.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(data):
    # Prepare features and target
    features = data[['SMA_10', 'SMA_20', 'SMA_50', 'RSI', 'MACD_Hist', 'ADX', '+DI', '-DI', 'Sentiment']]
    target = data['Final_Signal'].shift(-1)  # Predict next period's signal

    # Drop rows with missing values
    features = features.dropna()
    target = target.dropna()

    # Ensure features and target have the same index
    common_index = features.index.intersection(target.index)
    features = features.loc[common_index]
    target = target.loc[common_index]

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features, target)

    # Save model
    joblib.dump(model, 'ml_model.pkl')
    return model

def predict_signal(data, model):
    # Prepare features
    features = data[['SMA_10', 'SMA_20', 'SMA_50', 'RSI', 'MACD_Hist', 'ADX', '+DI', '-DI', 'Sentiment']]
    features = features.dropna()

    # Predict signals
    data.loc[features.index, 'ML_Signal'] = model.predict(features)
    return data