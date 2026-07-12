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
# # Feature Engineering
#
# In this notebook, I will create features for the machine learning models. I will extract time features, calculate lag variables, build rolling window statistics, and create weather metrics like Heating and Cooling Degree Days.
#
# Note: Since our data has multiple households, I must group by `LCLid` before shifting or calculating rolling averages so we do not mix values from different houses.

# %%
# Setup and imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc
import warnings
warnings.filterwarnings('ignore')

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
# ## Load Preprocessed Data
#
# I will load the cleaned master dataset and the UK holidays list.

# %%
# Get file paths
if IN_COLAB:
    from google.colab import files
    print("Upload master_daily.parquet")
    uploaded = files.upload()
    master_path = list(uploaded.keys())[0]
    
    print("\nUpload uk_bank_holidays.csv")
    uploaded_holidays = files.upload()
    holidays_path = list(uploaded_holidays.keys())[0]
else:
    master_path = '../data/processed/master_daily.parquet'
    holidays_path = '../data/raw/uk_bank_holidays.csv'
    
    # Check files
    for path in [master_path, holidays_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

print("File paths set.")

# %%
# Load files
print("Loading master dataset...")
df = pd.read_parquet(master_path)
print(f"Master shape: {df.shape}")

print("Loading holiday dataset...")
holidays_df = pd.read_csv(holidays_path)
print(f"Holidays shape: {holidays_df.shape}")

# %% [markdown]
# ## Calendar Features
#
# I will extract columns like month, day of week, weekend indicator, and check if the date is a UK bank holiday.

# %%
# Parse dates and sort
df['day'] = pd.to_datetime(df['day'])
df = df.sort_values(['LCLid', 'day']).reset_index(drop=True)

# Extract calendar columns
df['month'] = df['day'].dt.month.astype('int8')
df['day_of_week'] = df['day'].dt.dayofweek.astype('int8')
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype('int8')

# Define seasons
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'

df['season'] = df['month'].apply(get_season).astype('category')

# Merge holidays
holidays_df['Bank holidays'] = pd.to_datetime(holidays_df['Bank holidays'])
holiday_dates = set(holidays_df['Bank holidays'].dt.date)
df['is_holiday'] = df['day'].dt.date.isin(holiday_dates).astype('int8')

print("Calendar features created:")
print(df[['day', 'month', 'day_of_week', 'is_weekend', 'season', 'is_holiday']].head(3))

# %% [markdown]
# ## Weather Features (HDD & CDD)
#
# I will calculate Heating Degree Days (HDD) and Cooling Degree Days (CDD). These features measure how far the average daily temperature deviates from baseline comfort values.
# - HDD: `max(0, 15.5 - average_temp)` (indicates heating needs)
# - CDD: `max(0, average_temp - 22)` (indicates cooling needs)

# %%
# Calculate temperature variables
if 'temperatureMax' in df.columns and 'temperatureMin' in df.columns:
    df['temp_avg'] = ((df['temperatureMax'] + df['temperatureMin']) / 2).astype('float32')
    
    # HDD with 15.5 C baseline
    df['HDD'] = np.maximum(0, 15.5 - df['temp_avg']).astype('float32')
    
    # CDD with 22 C baseline
    df['CDD'] = np.maximum(0, df['temp_avg'] - 22.0).astype('float32')
    
    # Temperature range
    df['temp_range'] = (df['temperatureMax'] - df['temperatureMin']).astype('float32')

print("Weather features created:")
print(df[['temp_avg', 'HDD', 'CDD', 'temp_range']].describe().T)

# %% [markdown]
# ## Lag & Rolling Window Features
#
# I will create lag features (historical consumption values) and rolling window averages. To prevent target leakage, I will calculate rolling features using the lagged values (`energy_lag_1`) instead of the current day's target.

# %%
# Create lags and rolling features per household
target = 'energy_mean'

print("Calculating lags...")
df['energy_lag_1'] = df.groupby('LCLid')[target].shift(1).astype('float32')
df['energy_lag_7'] = df.groupby('LCLid')[target].shift(7).astype('float32')
df['energy_lag_14'] = df.groupby('LCLid')[target].shift(14).astype('float32')
df['energy_lag_28'] = df.groupby('LCLid')[target].shift(28).astype('float32')

print("Calculating rolling windows...")
df['energy_roll_mean_7'] = df.groupby('LCLid')['energy_lag_1'].transform(lambda x: x.rolling(7).mean()).astype('float32')
df['energy_roll_std_7'] = df.groupby('LCLid')['energy_lag_1'].transform(lambda x: x.rolling(7).std()).astype('float32')
df['energy_roll_mean_30'] = df.groupby('LCLid')['energy_lag_1'].transform(lambda x: x.rolling(30).mean()).astype('float32')

# %%
# Drop empty rows created by shifts/rolling windows
initial_len = len(df)
df = df.dropna(subset=['energy_lag_28', 'energy_roll_mean_30'])
print(f"Dropped {initial_len - len(df)} initial rows containing nulls.")

# %% [markdown]
# ### Observation
#
# Creating a 28-day lag and a 30-day rolling window means the first 30 days of records for each household contain missing values. Dropping these rows is necessary because the models need complete features to run predictions.

# %% [markdown]
# ## Categorical Encoding
#
# I will encode the tariff types and ACORN demographic groups as numeric codes.

# %%
# Encode Tariff
if 'stdorToU' in df.columns:
    df['tariff_code'] = df['stdorToU'].map({'Std': 0, 'ToU': 1}).astype('int8')

# Encode ACORN Groups
if 'Acorn' in df.columns:
    acorn_categories = sorted(df['Acorn'].unique())
    acorn_mapping = {cat: idx for idx, cat in enumerate(acorn_categories)}
    df['acorn_code'] = df['Acorn'].map(acorn_mapping).astype('int8')
    print("ACORN group mapping:")
    print(acorn_mapping)

# %% [markdown]
# ## Collinearity & Feature Selection
#
# I will compute the correlation matrix to check for highly redundant features.

# %%
# Correlation of numerical features
numeric_cols = [
    'energy_mean', 'energy_lag_1', 'energy_lag_7', 'energy_lag_14', 'energy_lag_28',
    'energy_roll_mean_7', 'energy_roll_std_7', 'energy_roll_mean_30',
    'temp_avg', 'HDD', 'CDD', 'temp_range', 'temperatureMax', 'temperatureHigh',
    'is_weekend', 'is_holiday', 'tariff_code', 'acorn_code'
]

corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'shrink': 0.8}, linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observation
#
# The correlation plot shows that `temperatureMax` and `temperatureHigh` are almost perfectly correlated (r = 1.0). I will drop `temperatureHigh` since it is redundant. I will also drop non-numeric text columns like `season` and `stdorToU` because they have been encoded.

# %%
# Drop redundant columns
features_to_drop = ['temperatureHigh', 'stdorToU', 'Acorn', 'season']
df_final = df.drop(columns=[col for col in features_to_drop if col in df.columns])

print(f"Final columns ({len(df_final.columns)}):\n{list(df_final.columns)}")

# %% [markdown]
# ## Save Feature Matrix
#
# I will save the engineered feature dataset to a Parquet file.

# %%
output_dir = '../data/processed'
features_parquet_path = os.path.join(output_dir, 'master_features.parquet')

# Save dataset
print("Saving feature matrix...")
df_final.to_parquet(features_parquet_path, index=False)
print(f"Saved: {features_parquet_path}")

final_mem = df_final.memory_usage(deep=True).sum() / 1024**2
print(f"Final Memory Usage: {final_mem:.2f} MB")
gc.collect()

# %% [markdown]
# ## Summary of Features
#
# - Extracted calendar features (`month`, `day_of_week`, `is_weekend`, `is_holiday`).
# - Generated thermodynamic indicators (`temp_avg`, `HDD`, `CDD`, `temp_range`).
# - Built lag variables (`energy_lag_1`, `energy_lag_7`, `energy_lag_14`, `energy_lag_28`) and rolling statistics (`energy_roll_mean_7`, `energy_roll_mean_30`) grouped by household to avoid cross-household data leakage.
# - Dropped initial rows containing nulls.
# - Encoded categorical demographic and tariff classes.
# - Removed redundant collinear weather columns to simplify model training.
