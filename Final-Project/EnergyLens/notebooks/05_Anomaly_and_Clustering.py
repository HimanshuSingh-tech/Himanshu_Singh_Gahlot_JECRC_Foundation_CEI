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
# # Anomaly Detection and Clustering
#
# In this notebook, I will work on two unsupervised tasks:
# 1. Anomaly Detection: Finding days of abnormal energy usage.
# 2. Household Clustering: Grouping households based on their consumption behavior.
#
# Note: Instead of clustering millions of daily rows directly, which is slow and memory-intensive, I will create a summary profile vector for each household (using average consumption, volatility, weekend bias, and winter/summer ratios) and cluster the households.

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

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

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
# I will load the final feature matrix.

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

# %% [markdown]
# ## Anomaly Detection
#
# I will compare two methods on a single household sample to see what they flag:
# - Rolling Z-Score: Flags days where energy is more than 3 standard deviations from the rolling average.
# - Isolation Forest: A machine learning method that isolates anomalies.

# %%
# Filter to a single household for testing
sample_household = df['LCLid'].iloc[0]
hh_df = df[df['LCLid'] == sample_household].sort_values('day').reset_index(drop=True)
print(f"Household: {sample_household} ({len(hh_df)} records)")

# 1. Rolling Z-Score Method
hh_df['z_score'] = (hh_df['energy_mean'] - hh_df['energy_roll_mean_7']) / (hh_df['energy_roll_std_7'] + 1e-5)
hh_df['anomaly_z'] = (np.abs(hh_df['z_score']) > 3.0).astype(int)

print(f"Rolling Z-Score anomalies: {hh_df['anomaly_z'].sum()}")

# %%
# 2. Isolation Forest Method
features_for_anomaly = ['energy_mean', 'energy_std', 'temp_avg', 'is_weekend']
iso_forest = IsolationForest(contamination=0.02, random_state=42)

# Fit model
hh_df['anomaly_if'] = iso_forest.fit_predict(hh_df[features_for_anomaly])
hh_df['anomaly_if'] = (hh_df['anomaly_if'] == -1).astype(int)

print(f"Isolation Forest anomalies: {hh_df['anomaly_if'].sum()}")

# %%
# Plot comparisons
plt.figure(figsize=(16, 6))
plt.plot(hh_df['day'], hh_df['energy_mean'], label='Daily Consumption', color='gray', alpha=0.6)

# Z-Score anomalies
z_anom = hh_df[hh_df['anomaly_z'] == 1]
plt.scatter(z_anom['day'], z_anom['energy_mean'], color='red', label='Z-Score Anomaly', marker='x', s=80)

# Isolation Forest anomalies
if_anom = hh_df[hh_df['anomaly_if'] == 1]
plt.scatter(if_anom['day'], if_anom['energy_mean'], color='blue', label='Isolation Forest Anomaly', facecolors='none', edgecolors='blue', s=120)

plt.title(f'Energy Consumption Anomalies for Household {sample_household}', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Energy Consumption (kWh)')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observation
#
# The Rolling Z-score flags spikes where energy increases suddenly compared to the surrounding week. Isolation Forest is a bit more flexible because it considers multiple variables, such as temperature, and can flag abnormally high or low points.

# %% [markdown]
# ## Household Profile Engineering
#
# I will aggregate the daily energy rows into a single summary profile per household.
#
# To avoid slow loop calculations in Python, I will use vectorized operations with pandas.
# - Winter-to-summer consumption ratio (thermal sensitivity)
# - Weekend-to-weekday ratio (weekend bias)
# - General consumption averages and standard deviations

# %%
print("Aggregating daily rows into household profiles...")

# Mask columns for season and day type calculations
df['winter_consumption'] = df['energy_mean'].where(df['month'].isin([12, 1, 2]))
df['summer_consumption'] = df['energy_mean'].where(df['month'].isin([6, 7, 8]))
df['weekend_consumption'] = df['energy_mean'].where(df['is_weekend'] == 1)
df['weekday_consumption'] = df['energy_mean'].where(df['is_weekend'] == 0)

# Group by household
profiles = df.groupby('LCLid').agg(
    mean_daily_consumption=('energy_mean', 'mean'),
    std_consumption=('energy_mean', 'std'),
    winter_mean=('winter_consumption', 'mean'),
    summer_mean=('summer_consumption', 'mean'),
    weekend_mean=('weekend_consumption', 'mean'),
    weekday_mean=('weekday_consumption', 'mean'),
    tariff_code=('tariff_code', 'first'),
    acorn_code=('acorn_code', 'first'),
    total_days=('energy_mean', 'count')
).reset_index()

# Clean up temp columns
df = df.drop(columns=['winter_consumption', 'summer_consumption', 'weekend_consumption', 'weekday_consumption'])

# Filter out households with low historical spans
profiles = profiles[profiles['total_days'] >= 100]
profiles = profiles.dropna(subset=['winter_mean', 'summer_mean'])

# Calculate ratios
profiles['thermal_sensitivity'] = profiles['winter_mean'] / (profiles['summer_mean'] + 1e-5)
# Cap extreme values to prevent outliers from creating one-household clusters
profiles['thermal_sensitivity'] = np.minimum(profiles['thermal_sensitivity'], 5.0)

profiles['weekend_bias'] = profiles['weekend_mean'] / (profiles['weekday_mean'] + 1e-5)

# Drop final null rows if any
profiles = profiles.dropna()
print(f"Generated profile vectors for {profiles.shape[0]} households.")
profiles.head(3)

# %% [markdown]
# ### Observation
#
# I noticed that one household previously had an extreme thermal sensitivity value, which skewed the scaling and caused KMeans to place that single household in its own cluster. Capping the thermal sensitivity feature at 5.0 prevents this outlier issue.

# %% [markdown]
# ## Finding the Optimal K (Elbow Method)
#
# I will use KMeans and test different values of K from 1 to 7 to find the optimal number of clusters.

# %%
# Select clustering features
cluster_features = ['mean_daily_consumption', 'std_consumption', 'thermal_sensitivity', 'weekend_bias']

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(profiles[cluster_features])

# Inertia test
inertia = []
K_range = range(1, 8)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# Plot Elbow Chart
plt.figure(figsize=(10, 5))
plt.plot(K_range, inertia, marker='o', color='#00d2ff')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observation
#
# The elbow plot shows a clear bend around K = 3. This means 3 clusters is a good choice for partitioning the households.

# %%
# Run KMeans with K=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
profiles['cluster'] = kmeans.fit_predict(scaled_features)
print("KMeans clustering completed.")

# %% [markdown]
# ## Cluster Archetypes
#
# I will check the mean values for each cluster to identify their energy usage profiles.

# %%
# Analyze cluster statistics
cluster_summary = profiles.groupby('cluster')[cluster_features].mean().reset_index()
print("Cluster Profiles:")
print("=" * 60)
print(cluster_summary.to_string(index=False))

# %% [markdown]
# ### Observation
#
# The statistics show three clear household archetypes:
# - **Cluster 0:** High daily consumption, high standard deviation, and higher thermal sensitivity. These are high-energy households that are highly sensitive to cold weather (likely electric heating).
# - **Cluster 1:** Low daily consumption and low thermal sensitivity. These are energy-efficient or small households.
# - **Cluster 2:** Average daily consumption and moderate thermal sensitivity. These represent typical households.

# %%
# Save cluster profile output
profiles_out_path = '../data/processed/household_profiles.csv'
profiles[['LCLid', 'cluster'] + cluster_features].to_csv(profiles_out_path, index=False)
print(f"Saved household profiles to: {profiles_out_path}")

# %% [markdown]
# ## Visualizing Clusters in PCA
#
# I will use PCA to reduce the 4 features to 2 dimensions for plotting.

# %%
# Run PCA
pca = PCA(n_components=2, random_state=42)
pca_features = pca.fit_transform(scaled_features)

profiles['pca_1'] = pca_features[:, 0]
profiles['pca_2'] = pca_features[:, 1]

# Scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=profiles, x='pca_1', y='pca_2', hue='cluster', 
    palette='viridis', style='cluster', s=50, alpha=0.8
)
plt.title('Household Segments projected in PCA 2D space', fontsize=14)
plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section Summary & Questions
#
# ### Q1: What is the benefit of Isolation Forest over rolling Z-score?
# Z-score only checks one variable at a time (univariate), meaning it can miss situations where a combination of features is abnormal. Isolation Forest evaluates multiple columns simultaneously, catching multivariate anomalies (e.g. low consumption on a very cold day).
#
# ### Q2: Why did we cluster households on profile metrics instead of daily rows?
# Clustering raw daily rows would assign the same household to different clusters depending on the season or day of the week. By aggregating daily data into household profiles, we segment the households themselves.
#
# ### Next Steps (Notebook 6):
# In the final notebook, I will build the forecasting pipeline using the XGBoost model, adding cluster features, and setting up recommendations.
