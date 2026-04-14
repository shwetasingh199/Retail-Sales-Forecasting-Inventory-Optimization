import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle
import os

# create folders if not exist
os.makedirs("models", exist_ok=True)

# load data
df = pd.read_csv("data/processed/feature_data.csv")

# features
features = [
    'lag_1','lag_7','lag_14',
    'rolling_mean_7','rolling_mean_14',
    'weekday'
]

X = df[features]
y = df['Sales']

# train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# save model
with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained & saved successfully!")