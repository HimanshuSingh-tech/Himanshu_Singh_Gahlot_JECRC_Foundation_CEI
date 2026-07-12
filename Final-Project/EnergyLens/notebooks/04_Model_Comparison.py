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
# # Model Comparison
#
# In this notebook, I will train and compare multiple forecasting models to predict domestic energy demand. 
#
# I will evaluate:
# 1. Linear Regression (Baseline)
# 2. Random Forest Regressor
# 3. XGBoost Regressor
# 4. Facebook Prophet
# 5. SARIMAX
#
# I will aggregate the data across all households to benchmark how these models perform on the total grid demand.

# %%
# Setup and imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Install Prophet if needed
try:
    from prophet import Prophet
except ImportError:
    print("Prophet not found. Installing now...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "prophet", "--quiet"])
    from prophet import Prophet

sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams['figure.figsize'] = (14, 6)

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
# I will load the feature matrix file produced in Notebook 3.

# %%
# Get file path
if IN_COLAB:
    from google.colab import files
    print("Upload master_features.parquet")
    uploaded = files.upload()
    features_path = list(uploaded.keys())[0]
else:
    features_path = '../data/processed/master_features.parquet'
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing file: {features_path}")

# Load the dataset
print("Loading feature matrix...")
df = pd.read_parquet(features_path)
print(f"Dataset shape: {df.shape}")
df.head(3)

# %% [markdown]
# ## Chronological Train/Test Split
#
# Since this is time-series data, I cannot split the training and testing sets randomly. Doing so would lead to future information leaking into the past. I will split the data chronologically:
# - Train: November 2011 to December 2013
# - Test: January 2014 to February 2014

# %%
# Split data chronologically
df['day'] = pd.to_datetime(df['day'])
split_date = pd.to_datetime('2014-01-01')

print(f"Training range: {df['day'].min()} to {split_date - pd.to_timedelta('1D')}")
print(f"Testing range: {split_date} to {df['day'].max()}")

# %% [markdown]
# ## Grid Demand Benchmarking
#
# I will group the data by date and calculate average consumption and features across all households. This gives a single daily energy load curve representing the total London grid demand.

# %%
# Group by day
agg_df = df.groupby('day').agg({
    'energy_mean': 'mean',
    'energy_lag_1': 'mean',
    'energy_lag_7': 'mean',
    'energy_lag_14': 'mean',
    'energy_lag_28': 'mean',
    'energy_roll_mean_7': 'mean',
    'energy_roll_std_7': 'mean',
    'energy_roll_mean_30': 'mean',
    'temp_avg': 'mean',
    'HDD': 'mean',
    'CDD': 'mean',
    'temp_range': 'mean',
    'is_weekend': 'first',
    'is_holiday': 'first'
}).reset_index()

agg_df = agg_df.sort_values('day').reset_index(drop=True)
print(f"Aggregate shape: {agg_df.shape}")

# Split aggregate dataset into train and test
train_agg = agg_df[agg_df['day'] < split_date]
test_agg = agg_df[agg_df['day'] >= split_date]

print(f"Train Shape: {train_agg.shape}")
print(f"Test Shape: {test_agg.shape}")

# Define feature columns
feature_cols = [
    'energy_lag_1', 'energy_lag_7', 'energy_lag_14', 'energy_lag_28',
    'energy_roll_mean_7', 'energy_roll_std_7', 'energy_roll_mean_30',
    'temp_avg', 'HDD', 'CDD', 'temp_range', 'is_weekend', 'is_holiday'
]

# Separate features (X) and target (y)
X_train_agg, y_train_agg = train_agg[feature_cols], train_agg['energy_mean']
X_test_agg, y_test_agg = test_agg[feature_cols], test_agg['energy_mean']

# %% [markdown]
# ## Model Evaluation Helper
#
# I will write a small helper function to calculate metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and R-squared.

# %%
# Performance log
results_dict = {}

def evaluate_predictions(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    results_dict[model_name] = {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'R²': r2
    }
    
    print(f"{model_name} Performance:")
    print(f"  MAE:  {mae:.4f} kWh")
    print(f"  RMSE: {rmse:.4f} kWh")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}\n")
    return y_pred

# %% [markdown]
# ## Model Training & Forecasting
#
# I will train each of the five models and record their metrics on the test split.

# %%
# 1. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_agg, y_train_agg)
lr_preds = lr_model.predict(X_test_agg)
evaluate_predictions(y_test_agg, lr_preds, "Linear Regression")

# %%
# 2. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train_agg, y_train_agg)
rf_preds = rf_model.predict(X_test_agg)
evaluate_predictions(y_test_agg, rf_preds, "Random Forest")

# %%
# 3. XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
xgb_model.fit(X_train_agg, y_train_agg)
xgb_preds = xgb_model.predict(X_test_agg)
evaluate_predictions(y_test_agg, xgb_preds, "XGBoost")

# %%
# 4. Facebook Prophet
# Format data for Prophet
prophet_train = train_agg[['day', 'energy_mean']].rename(columns={'day': 'ds', 'energy_mean': 'y'})
prophet_train['temp_avg'] = train_agg['temp_avg']
prophet_train['is_weekend'] = train_agg['is_weekend']

# Fit model
m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
m.add_regressor('temp_avg')
m.add_regressor('is_weekend')
m.fit(prophet_train)

# Predict test set
prophet_test = test_agg[['day']].rename(columns={'day': 'ds'})
prophet_test['temp_avg'] = test_agg['temp_avg']
prophet_test['is_weekend'] = test_agg['is_weekend']

prophet_forecast = m.predict(prophet_test)
prophet_preds = prophet_forecast['yhat'].values
evaluate_predictions(y_test_agg, prophet_preds, "Facebook Prophet")

# %%
# 5. SARIMAX
# Define variables
endog_train = train_agg['energy_mean']
exog_train = train_agg[['temp_avg', 'is_weekend']]

# Fit SARIMAX
sarimax_model = SARIMAX(endog_train, exog=exog_train, order=(1, 1, 1), seasonal_order=(1, 0, 0, 7))
sarimax_results = sarimax_model.fit(disp=False)

exog_test = test_agg[['temp_avg', 'is_weekend']]
sarimax_preds = sarimax_results.forecast(steps=len(test_agg), exog=exog_test).values
evaluate_predictions(y_test_agg, sarimax_preds, "SARIMAX")

# %% [markdown]
# ## Performance Comparison
#
# I will build a comparison table and plot all model forecasts against the actual values.

# %%
# Print table
comparison_df = pd.DataFrame(results_dict).T
print("Model Comparison Summary:")
print("=" * 60)
print(comparison_df.sort_values('MAE').to_string())

# %%
# Plot comparisons
plt.figure(figsize=(16, 8))
plt.plot(test_agg['day'], y_test_agg, label='Actual Demand', color='black', linewidth=2.5)
plt.plot(test_agg['day'], lr_preds, label='Linear Regression', linestyle='--', alpha=0.7)
plt.plot(test_agg['day'], rf_preds, label='Random Forest', linestyle='--', alpha=0.7)
plt.plot(test_agg['day'], xgb_preds, label='XGBoost', color='cyan', linewidth=2)
plt.plot(test_agg['day'], prophet_preds, label='Facebook Prophet', color='magenta', linewidth=2)
plt.plot(test_agg['day'], sarimax_preds, label='SARIMAX', linestyle='--', alpha=0.7)

plt.title('London Grid Energy Demand Forecast vs Actuals (Jan - Feb 2014)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Average Household Energy Consumption (kWh)', fontsize=12)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observation
#
# Comparing the models, tree-based regressors (XGBoost and Random Forest) achieved the lowest RMSE and highest R-squared values on the test set. Linear Regression served as a good baseline but struggles to capture complex temperature dependencies. Prophet and SARIMAX are strong for general seasonality but did not outperform XGBoost here.

# %% [markdown]
# ## Section Summary & Questions
#
# ### Q1: Why does XGBoost perform better than SARIMA on this data?
# XGBoost is a tree ensemble that handles nonlinear relationships (like cold temperatures causing exponential heating spikes) and can use custom engineered features (lags, rolling averages, calendars) directly. SARIMA is a linear model that relies strictly on its mathematical orders.
#
# ### Q2: Why did we build lag and rolling features on a 1-day shift instead of the current day?
# If we include the current day's value in rolling windows or lags, the model would be using the target variable itself to make the forecast. At prediction time, today's value is unknown. Shifting by 1 day ensures no future data is leaked.
#
# ### Q3: How do Heating Degree Days help the models?
# Energy consumption does not scale linearly with temperature. People only turn on heaters when it gets cold (under 15.5°C). HDDs translate temperature into a metric that starts at 0 and grows when temperature drops, representing this heating threshold.
#
# ### Next Steps (Notebook 5):
# I will work on unsupervised tasks:
# 1. Evaluate methods for detecting abnormal consumption spikes.
# 2. Cluster households into demographic groups using KMeans.
