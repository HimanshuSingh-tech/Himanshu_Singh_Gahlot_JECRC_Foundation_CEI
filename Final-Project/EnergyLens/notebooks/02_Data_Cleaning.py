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
# # Data Cleaning
#
# In this notebook, I will clean the datasets and merge them into a single file. I need to handle missing values, treat outliers, fix column formats, and optimize memory usage since the combined dataset will be large.

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
# ## Data Loading
#
# I will load the household info, daily weather, and daily energy consumption data.

# %%
# Map paths
if IN_COLAB:
    from google.colab import files
    print("Upload informations_households.csv")
    uploaded_hh = files.upload()
    hh_path = list(uploaded_hh.keys())[0]

    print("\nUpload weather_daily_darksky.csv")
    uploaded_weather = files.upload()
    weather_path = list(uploaded_weather.keys())[0]

    print("\nUpload daily_dataset.csv")
    uploaded_daily = files.upload()
    daily_path = list(uploaded_daily.keys())[0]
else:
    hh_path = '../data/raw/informations_households.csv'
    weather_path = '../data/raw/weather_daily_darksky.csv'
    daily_path = '../data/raw/daily_dataset.csv'
    
    # Check if files exist
    for path in [hh_path, weather_path, daily_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

print("File paths set.")

# %%
# Load files
print("Loading households data...")
households = pd.read_csv(hh_path)
print("Loading weather data...")
weather = pd.read_csv(weather_path)
print("Loading daily energy data...")
daily = pd.read_csv(daily_path)

print(f"\nInitial shapes:")
print(f"  Households: {households.shape}")
print(f"  Weather: {weather.shape}")
print(f"  Daily: {daily.shape}")

# %% [markdown]
# ## Missing Values
#
# I will look at missing values in each dataset and decide how to fill or remove them.

# %%
# Household missing values
print("Households null check:")
print(households.isnull().sum())

# Drop rows missing tariff or ACORN group as they are needed for household segmentation
initial_hh_count = len(households)
households = households.dropna(subset=['stdorToU', 'Acorn'])
print(f"Dropped {initial_hh_count - len(households)} households with missing demographics.")

# %%
# Weather missing values
print("Weather null check:")
print(weather.isnull().sum().sort_values(ascending=False).head(5))

# Fill missing weather columns using linear interpolation since weather is continuous
weather_cols_to_fill = ['cloudCover', 'uvIndex', 'uvIndexTime']
for col in weather_cols_to_fill:
    if col in weather.columns:
        null_count = weather[col].isnull().sum()
        if null_count > 0:
            weather[col] = weather[col].interpolate(method='linear').ffill().bfill()
            print(f"  Filled {null_count} missing values in '{col}' using linear interpolation.")

# Double check weather nulls
assert weather.isnull().sum().sum() == 0, "Weather still has missing values!"
print("Weather data is clean.")

# %%
# Daily consumption missing values
print("Daily energy null check:")
print(daily.isnull().sum())

# Check empty energy values
all_null_energy = daily[daily['energy_mean'].isnull()]
print(f"\nRows with missing energy_mean: {len(all_null_energy)}")

# Drop rows where energy_mean is null since there is no consumption data
daily = daily.dropna(subset=['energy_mean'])
print(f"Dropped {len(all_null_energy)} rows with missing energy records.")

# Check nulls again
print(f"Remaining nulls:")
print(daily.isnull().sum())

# %% [markdown]
# ### Observation
#
# Some rows in the daily dataset are missing the standard deviation (`energy_std`). I need to check why this happens before filling them.

# %%
# Handle energy_std missing values
std_null_count = daily['energy_std'].isnull().sum()
if std_null_count > 0:
    # See if energy_count is 1 when energy_std is null
    sample_null_std = daily[daily['energy_std'].isnull()]['energy_count'].value_counts()
    print("Value counts of energy_count when energy_std is null:")
    print(sample_null_std)
    
    # If count is 1, standard deviation is undefined. I will fill it with 0.0
    daily['energy_std'] = daily['energy_std'].fillna(0.0)
    print(f"Filled {std_null_count} missing values in 'energy_std' with 0.0")

# Verify daily is clean
assert daily.isnull().sum().sum() == 0, "Daily consumption still has missing values!"
print("Daily energy data is clean.")

# %% [markdown]
# ## Outlier Treatment
#
# I will check for outliers in energy consumption. I can use IQR or Z-score thresholds to see what they flag.

# %%
# Outlier detection on energy_mean
energy_data = daily['energy_mean']

# IQR method
Q1 = energy_data.quantile(0.25)
Q3 = energy_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound_iqr = Q1 - 1.5 * IQR
upper_bound_iqr = Q3 + 1.5 * IQR

# Z-score method (3 standard deviations)
mean_val = energy_data.mean()
std_val = energy_data.std()
upper_bound_z = mean_val + 3 * std_val
lower_bound_z = mean_val - 3 * std_val

print(f"Outlier Thresholds (energy_mean):")
print(f"  IQR bounds: {lower_bound_iqr:.4f} to {upper_bound_iqr:.4f}")
print(f"  Z-score bounds: {lower_bound_z:.4f} to {upper_bound_z:.4f}")

# Outliers counts
outliers_iqr = (energy_data < lower_bound_iqr) | (energy_data > upper_bound_iqr)
outliers_z = (energy_data < lower_bound_z) | (energy_data > upper_bound_z)

print(f"\n  IQR Outliers: {outliers_iqr.sum()} ({outliers_iqr.mean()*100:.2f}%)")
print(f"  Z-score Outliers: {outliers_z.sum()} ({outliers_z.mean()*100:.2f}%)")

# %% [markdown]
# ### Observation
#
# The IQR method flags more than 10% of the data as outliers, which is too high because daily consumption has a long right tail. Instead of dropping rows or using IQR bounds, I will cap extreme values at the 99.5th percentile for each household. This keeps high-consuming households distinct without keeping extreme measurement spikes.

# %%
# Cap energy values at 99.5th percentile per household
print("Capping outliers at household-level 99.5th percentile...")

percentiles = daily.groupby('LCLid')['energy_mean'].transform(lambda x: x.quantile(0.995))
num_capped = (daily['energy_mean'] > percentiles).sum()
daily['energy_mean'] = np.minimum(daily['energy_mean'], percentiles)

print(f"Capped {num_capped} extreme values.")

# %% [markdown]
# ## Datetime Formatting
#
# I will convert the date columns to datetime objects to ensure they merge correctly.

# %%
# Convert dates
daily['day'] = pd.to_datetime(daily['day'])
print(f"Daily date range: {daily['day'].min()} to {daily['day'].max()}")

weather['time'] = pd.to_datetime(weather['time']).dt.normalize()
print(f"Weather date range: {weather['time'].min()} to {weather['time'].max()}")

# Drop duplicate dates in weather if any
weather = weather.drop_duplicates(subset=['time'])

# %% [markdown]
# ## Merging Datasets
#
# I will merge the daily consumption data with the household demographics and weather datasets.

# %%
# Merge daily and households
print("Merging daily consumption and households...")
master_df = daily.merge(households, on='LCLid', how='inner')
print(f"  Shape after household merge: {master_df.shape}")

# Merge with weather
print("Merging with weather data...")
master_df = master_df.merge(weather, left_on='day', right_on='time', how='inner')
master_df = master_df.drop(columns=['time'])
print(f"  Final merged shape: {master_df.shape}")

# %% [markdown]
# ## Memory Optimization
#
# Since the merged dataset is large, I will downcast data types to reduce memory usage in python.

# %%
# Initial memory usage
initial_mem = master_df.memory_usage(deep=True).sum() / 1024**2
print(f"Initial Memory: {initial_mem:.2f} MB")

# Downcast numerical columns and convert strings to categories
for col in master_df.columns:
    col_type = master_df[col].dtype
    
    if col_type == 'float64':
        master_df[col] = master_df[col].astype('float32')
        
    elif col_type == 'int64':
        max_val = master_df[col].max()
        min_val = master_df[col].min()
        if max_val < 32767 and min_val > -32768:
            master_df[col] = master_df[col].astype('int16')
        else:
            master_df[col] = master_df[col].astype('int32')
            
    elif col_type == 'object' and col not in ['day']:
        master_df[col] = master_df[col].astype('category')

# Final memory usage
optimized_mem = master_df.memory_usage(deep=True).sum() / 1024**2
savings = (initial_mem - optimized_mem) / initial_mem * 100
print(f"Optimized Memory: {optimized_mem:.2f} MB")
print(f"Memory Saved: {savings:.2f}%")

gc.collect()

# %%
# Show processed data
master_df.head(3)

# %% [markdown]
# ## Saving the Cleaned Dataset
#
# I will save the cleaned and merged dataset. I will save it as both Parquet (for efficient, fast loading in downstream notebooks) and a compressed CSV file.

# %%
output_dir = '../data/processed'
os.makedirs(output_dir, exist_ok=True)

parquet_path = os.path.join(output_dir, 'master_daily.parquet')
csv_path = os.path.join(output_dir, 'master_daily.csv')

# Save files
print("Saving Parquet...")
master_df.to_parquet(parquet_path, index=False)
print(f"Saved: {parquet_path}")

print("Saving CSV (zipped)...")
master_df.to_csv(csv_path + '.gz', index=False, compression='gzip')
print(f"Saved: {csv_path}.gz")

# Print disk size comparison
parquet_size = os.path.getsize(parquet_path) / 1024**2
csv_size = os.path.getsize(csv_path + '.gz') / 1024**2
print(f"\nFile Size Comparison:")
print(f"  Parquet: {parquet_size:.2f} MB")
print(f"  Compressed CSV: {csv_size:.2f} MB")

# %% [markdown]
# ## Summary of Cleaning
#
# - Handled missing values by dropping rows without demographics or energy records.
# - Filled missing standard deviation values (`energy_std`) with 0.0 when daily count was 1.
# - Linear interpolated missing weather columns.
# - Capped extreme outlier values at the 99.5th percentile per household.
# - Merged consumption, weather, and household metadata.
# - Downcasted datatypes, reducing memory usage significantly.
# - Saved the final dataset to a Parquet file for fast loading.
