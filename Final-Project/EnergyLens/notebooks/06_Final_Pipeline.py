# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Final Pipeline
#
# In this notebook, I will build the final pipeline. I will:
# 1. Train the final XGBoost forecasting model on the full feature matrix.
# 2. Train the KMeans clustering model and scaler on the household profile vectors.
# 3. Save these model files (using joblib) so they can be loaded by the web app.
# 4. Write a python class `EnergyLensPipeline` that coordinates the prediction flow: loading a household's profile, generating its 7-day energy forecast, flagging any anomalous days, and providing personalized energy conservation tips.

# %%
# Setup and imports
import pandas as pd
import numpy as np
import os
import gc
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib

# Check environment
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

print(f"Running in Colab: {IN_COLAB}")

# %% [markdown]
# ## Data Loading
#
# I will load the final features and household profiles.

# %%
# Get file paths
if IN_COLAB:
    from google.colab import files
    print("Upload master_features.parquet")
    uploaded_feats = files.upload()
    features_path = list(uploaded_feats.keys())[0]
    
    print("\nUpload household_profiles.csv")
    uploaded_profiles = files.upload()
    profiles_path = list(uploaded_profiles.keys())[0]
else:
    features_path = '../data/processed/master_features.parquet'
    profiles_path = '../data/processed/household_profiles.csv'
    
    # Check files
    for path in [features_path, profiles_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

# Load the datasets
print("Loading feature matrix...")
df = pd.read_parquet(features_path)
print("Loading household profiles...")
profiles = pd.read_csv(profiles_path)

print("Data loaded successfully.")

# %% [markdown]
# ## Model Training & Serialization
#
# I will train the final forecaster, scaler, and clusterer, and then save them to the `models/` directory.

# %%
# Create models folder
models_dir = '../models'
os.makedirs(models_dir, exist_ok=True)

# 1. Train and save XGBoost model
target = 'energy_mean'
features = [
    'energy_lag_1', 'energy_lag_7', 'energy_lag_14', 'energy_lag_28',
    'energy_roll_mean_7', 'energy_roll_std_7', 'energy_roll_mean_30',
    'temp_avg', 'HDD', 'CDD', 'temp_range', 'is_weekend', 'is_holiday',
    'tariff_code', 'acorn_code'
]

print("Training final XGBoost model...")
X = df[features]
y = df[target]

xgb_forecaster = xgb.XGBRegressor(n_estimators=120, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
xgb_forecaster.fit(X, y)

forecaster_path = os.path.join(models_dir, 'xgboost_forecaster.joblib')
joblib.dump(xgb_forecaster, forecaster_path)
print(f"Saved forecaster to: {forecaster_path}")

# %%
# 2. Train and save Scaler and KMeans
cluster_features = ['mean_daily_consumption', 'std_consumption', 'thermal_sensitivity', 'weekend_bias']

print("Training Scaler and KMeans...")
scaler = StandardScaler()
scaled_X = scaler.fit_transform(profiles[cluster_features])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(scaled_X)

scaler_path = os.path.join(models_dir, 'scaler.joblib')
clusterer_path = os.path.join(models_dir, 'kmeans_clusterer.joblib')

joblib.dump(scaler, scaler_path)
joblib.dump(kmeans, clusterer_path)

print(f"Saved scaler to: {scaler_path}")
print(f"Saved clusterer to: {clusterer_path}")

# %% [markdown]
# ## Smart Recommendation Rules
#
# I will define some simple recommendations based on the household's cluster:
# - Cluster 0: High heating sensitivity. Suggest check-up of radiator settings, window sealing, and insulation.
# - Cluster 1: Low-usage/efficient homes. Suggest ToU (Time-of-Use) shift opportunities to save money.
# - Cluster 2: Weekend-focused consumption. Suggest shifting high-energy usage tasks to weekend off-peaks.

# %%
# Recommendation function
def get_recommendations(cluster_id):
    recommendations = {
        0: [
            "🔥 Segment Archetype: Thermal/Heating Sensitive Home",
            "📊 Observation: Your energy usage spikes dramatically on cold winter days, suggesting electric heating.",
            "🛠️ Conservation Strategy: Inspect home insulation, loft seals, and window glazing. A smart thermostat (e.g. Nest, Hive) can save up to 15% on heating costs.",
            "🔌 Action: Set thermostatic radiator valves (TRVs) to heat occupied rooms only."
        ],
        1: [
            "💡 Segment Archetype: Frugal & Efficient Home",
            "📊 Observation: Your base energy usage is low and highly efficient.",
            "🛠️ Tariff Strategy: Switch to a ToU (Time-of-Use) tariff if you aren't on one. Shift laundry/dishwashing to late nights or early mornings to buy electricity at ultra-cheap off-peak rates.",
            "🔌 Action: Automate smart appliances to run before 16:00 or after 19:00."
        ],
        2: [
            "⚖️ Segment Archetype: Weekend Centric Consumption Profile",
            "📊 Observation: Your energy consumption is significantly higher on weekends than weekdays (weekend bias > 1.2).",
            "🛠️ Shifting Strategy: You are an excellent candidate for weekend-incentive tariffs. You can save substantially by shifting high-power tasks (e.g. washing, EV charging, deep cleaning) entirely to weekend daytime hours.",
            "🔌 Action: Use smart schedules to delay heavy appliance operations to Saturday/Sunday mornings and afternoons."
        ]
    }
    return recommendations.get(cluster_id, ["No recommendations available."])

# %% [markdown]
# ## End-to-End Pipeline Class
#
# I will write a unified class that takes a household ID, historical values, and forecast weather, then outputs predictions, anomaly alerts, and recommendations.

# %%
# Pipeline class
class EnergyLensPipeline:
    def __init__(self, models_dir='../models', profiles_path='../data/processed/household_profiles.csv'):
        # Load models
        self.forecaster = joblib.load(os.path.join(models_dir, 'xgboost_forecaster.joblib'))
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.joblib'))
        self.clusterer = joblib.load(os.path.join(models_dir, 'kmeans_clusterer.joblib'))
        
        # Load profiles
        self.profiles = pd.read_csv(profiles_path)
        self.profiles_dict = self.profiles.set_index('LCLid').to_dict(orient='index')
        
    def forecast_and_analyze(self, household_id, recent_history_df, weather_forecast_df):
        # 1. Profile Lookup
        if household_id not in self.profiles_dict:
            return {"Error": f"Household {household_id} not found in database."}
        
        hh_profile = self.profiles_dict[household_id]
        cluster = hh_profile['cluster']
        
        # 2. Build Future Feature Matrix for Forecasting
        inf_df = weather_forecast_df.copy()
        inf_df['LCLid'] = household_id
        inf_df['tariff_code'] = hh_profile.get('tariff_code', 0)
        inf_df['acorn_code'] = hh_profile.get('acorn_code', 5)
        
        # Fill lags/rolling features from the last values of recent_history_df
        last_val = recent_history_df['energy_mean'].iloc[-1]
        last_mean_7 = recent_history_df['energy_mean'].iloc[-7:].mean()
        last_std_7 = recent_history_df['energy_mean'].iloc[-7:].std()
        last_mean_30 = recent_history_df['energy_mean'].mean()
        
        # Set lag approximations for the 7-day horizon
        inf_df['energy_lag_1'] = last_val
        inf_df['energy_lag_7'] = recent_history_df['energy_mean'].iloc[-7] if len(recent_history_df) >= 7 else last_val
        inf_df['energy_lag_14'] = recent_history_df['energy_mean'].iloc[-14] if len(recent_history_df) >= 14 else last_val
        inf_df['energy_lag_28'] = recent_history_df['energy_mean'].iloc[-28] if len(recent_history_df) >= 28 else last_val
        
        inf_df['energy_roll_mean_7'] = last_mean_7
        inf_df['energy_roll_std_7'] = last_std_7
        inf_df['energy_roll_mean_30'] = last_mean_30
        
        features_order = [
            'energy_lag_1', 'energy_lag_7', 'energy_lag_14', 'energy_lag_28',
            'energy_roll_mean_7', 'energy_roll_std_7', 'energy_roll_mean_30',
            'temp_avg', 'HDD', 'CDD', 'temp_range', 'is_weekend', 'is_holiday',
            'tariff_code', 'acorn_code'
        ]
        
        # 3. Forecast Next 7 Days
        X_inf = inf_df[features_order]
        forecasts = self.forecaster.predict(X_inf)
        
        # 4. Anomaly Warning Check
        anomaly_threshold = hh_profile['mean_daily_consumption'] + 2 * hh_profile['std_consumption']
        anomalous_days = []
        for i, val in enumerate(forecasts):
            if val > anomaly_threshold:
                anomalous_days.append(str(inf_df['day'].iloc[i].date()))
                
        # 5. Compile Recommendations
        recs = get_recommendations(cluster)
        
        return {
            "household_id": household_id,
            "cluster_id": cluster,
            "cluster_archetype": recs[0].split(': ')[1],
            "7_day_forecast": list(forecasts.round(3)),
            "forecast_dates": [str(d.date()) for d in inf_df['day']],
            "anomalous_days_detected": anomalous_days,
            "personalized_recommendations": recs[1:]
        }

# %% [markdown]
# ## Pipeline Demonstration
#
# I will run a sample household through the pipeline to test its outputs.

# %%
# Instantiate pipeline
pipeline = EnergyLensPipeline()

# Select a sample household
sample_hh = profiles['LCLid'].iloc[0]

# Simulate 30 days of recent history for this household
recent_history = df[df['LCLid'] == sample_hh].sort_values('day').tail(30)

# Simulate a 7-day upcoming weather forecast
upcoming_dates = pd.date_range(start='2014-03-01', periods=7, freq='D')
weather_forecast = pd.DataFrame({
    'day': upcoming_dates,
    'temp_avg': [6.5, 5.0, 4.2, 5.5, 7.0, 8.2, 6.0],
    'HDD': [9.0, 10.5, 11.3, 10.0, 8.5, 7.3, 9.5],
    'CDD': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    'temp_range': [4.0, 5.0, 3.5, 4.0, 5.5, 6.0, 4.5],
    'is_weekend': [1, 1, 0, 0, 0, 0, 0],
    'is_holiday': [0, 0, 0, 0, 0, 0, 0]
})

# Run inference
output = pipeline.forecast_and_analyze(sample_hh, recent_history, weather_forecast)

# Print final pipeline output
import json
print(json.dumps(output, indent=2))

# %% [markdown]
# ## Summary of Pipeline
#
# - Finalized the XGBoost forecaster and KMeans clustering models.
# - Serialized all model components and scaling assets to `.joblib` files.
# - Created a recommendations lookup based on household cluster categories.
# - Implemented the `EnergyLensPipeline` class to handle end-to-end data retrieval, feature scaling, 7-day forecasting, and anomaly checking.
# - Successfully tested the class on a sample household.
