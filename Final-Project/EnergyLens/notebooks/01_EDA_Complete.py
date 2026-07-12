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
# # Data Exploration
#
# In this notebook, I will explore the datasets for the energy forecasting project. I want to check the structure, data types, missing values, and visual patterns of our datasets to understand the relationships before building any models.

# %%
# Setup and imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="darkgrid", palette="viridis")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12

print("Libraries loaded successfully")

# %% [markdown]
# ## Household Information Exploration
#
# I will load and check the household details file, which contains information like demographics (ACORN groups) and tariff types.

# %%
# Upload household file
from google.colab import files
print("Upload: informations_households.csv")
uploaded_hh = files.upload()

# %%
# Load household data
hh_filename = list(uploaded_hh.keys())[0]
households = pd.read_csv(hh_filename)

print(f"Shape: {households.shape}")
print(f"\nColumns: {list(households.columns)}")
print(f"\nData types:\n{households.dtypes}")
print(f"\nFirst 5 rows:")
households.head()

# %% [markdown]
# ### Observation
#
# The dataset has information for individual households, including their tariff type (standard or time of use) and their ACORN demographic classification.

# %%
# Household metadata summary
print("=" * 60)
print("HOUSEHOLD METADATA SUMMARY")
print("=" * 60)
print(f"\nTotal households: {households.shape[0]}")

if 'Acorn' in households.columns:
    print(f"\nUnique ACORN groups: {households['Acorn'].nunique()}")

if 'stdorToU' in households.columns:
    print(f"\nTariff types:")
    print(households['stdorToU'].value_counts().to_string())

print(f"\nMissing values:\n{households.isnull().sum()}")

# %%
# Visualizing ACORN and Tariff distributions
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

acorn_col = 'Acorn_grouped' if 'Acorn_grouped' in households.columns else 'Acorn'
if acorn_col in households.columns:
    acorn_counts = households[acorn_col].value_counts()
    axes[0].barh(acorn_counts.index, acorn_counts.values, color='#00d2ff')
    axes[0].set_title(f'Households by {acorn_col}')
    axes[0].set_xlabel('Count')

if 'stdorToU' in households.columns:
    tariff_counts = households['stdorToU'].value_counts()
    axes[1].pie(tariff_counts.values, labels=tariff_counts.index,
                autopct='%1.1f%%', colors=['#00d2ff', '#ff6b6b'])
    axes[1].set_title('Tariff Type Distribution')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Observation
#
# Most households in the sample use standard tariffs, with only a small percentage using time-of-use (ToU). The horizontal bar chart shows how households are distributed across different ACORN demographic groups.

# %% [markdown]
# ## Weather Data Exploration
#
# I will inspect the weather dataset to see temperature variables and other weather conditions that might affect energy consumption.

# %%
# Upload weather data
print("Upload: weather_daily_darksky.csv")
uploaded_weather = files.upload()

# %%
weather_filename = list(uploaded_weather.keys())[0]
weather_daily = pd.read_csv(weather_filename)

print(f"Shape: {weather_daily.shape}")
print(f"\nColumns ({len(weather_daily.columns)}):\n{list(weather_daily.columns)}")
print(f"\nData types:\n{weather_daily.dtypes}")
print(f"\nMissing values (top 10):\n{weather_daily.isnull().sum().sort_values(ascending=False).head(10)}")
print("\nBasic statistics:")
weather_daily.describe()

# %%
# Check date column range
date_col = [c for c in weather_daily.columns if 'time' in c.lower() and 'max' not in c.lower() and 'min' not in c.lower() and 'high' not in c.lower() and 'low' not in c.lower() and 'sunrise' not in c.lower() and 'sunset' not in c.lower() and 'uv' not in c.lower() and 'apparent' not in c.lower()]
if not date_col:
    date_col = [c for c in weather_daily.columns if c.lower() == 'time']
if date_col:
    weather_daily[date_col[0]] = pd.to_datetime(weather_daily[date_col[0]])
    print(f"Date column: {date_col[0]}")
    print(f"Date range: {weather_daily[date_col[0]].min()} to {weather_daily[date_col[0]].max()}")
    print(f"Total days: {(weather_daily[date_col[0]].max() - weather_daily[date_col[0]].min()).days}")
else:
    print("No date column found automatically.")

# %%
# Plot temperature trend
if date_col:
    temp_cols = [c for c in weather_daily.columns if 'temperaturemax' == c.lower()]
    if not temp_cols:
        temp_cols = [c for c in weather_daily.columns if 'temp' in c.lower() and weather_daily[c].dtype in ['float64','int64']]
    if temp_cols:
        fig = px.line(weather_daily.sort_values(date_col[0]), x=date_col[0], y=temp_cols[0],
                      title='Daily Temperature Trend — London',
                      labels={date_col[0]: 'Date', temp_cols[0]: 'Temperature (°C)'})
        fig.update_layout(template='plotly_dark')
        fig.show()

# %% [markdown]
# ### Observation
#
# The temperature plot shows a clear annual cycle, with cold temperatures during winter (usually around January-February) and warm temperatures during summer (July-August).

# %% [markdown]
# ## Daily Consumption Exploration
#
# I will load the daily aggregated energy usage data per household to examine general patterns.

# %%
# Upload daily energy consumption files
print("Upload: daily_dataset.csv")
uploaded_daily = files.upload()

# %%
# Load and combine daily consumption datasets
daily_frames = []
for fname, content in uploaded_daily.items():
    print(f"Loading: {fname}")
    df = pd.read_csv(fname)
    daily_frames.append(df)
    print(f"  Shape: {df.shape}")

daily = pd.concat(daily_frames, ignore_index=True)
print(f"\nCombined daily dataset shape: {daily.shape}")
print(f"Columns: {list(daily.columns)}")
print(f"\nData types:\n{daily.dtypes}")
print(f"\nMissing values:\n{daily.isnull().sum()}")
print(f"\nFirst 5 rows:")
daily.head()

# %%
# Statistical summary of consumption
numeric_cols = daily.select_dtypes(include=[np.number]).columns.tolist()
print("Daily Consumption — Statistical Summary:")
print("=" * 60)
print(daily[numeric_cols].describe())
print(f"\nUnique households: {daily.iloc[:, 0].nunique()}")

# %% [markdown]
# ## Consumption Distribution Analysis
#
# I will plot the distribution of energy values to check for skewness.

# %%
# Check distribution of energy values
energy_cols = [c for c in daily.columns if any(
    kw in c.lower() for kw in ['energy', 'kwh', 'consumption', 'sum', 'mean']
)]
print(f"Potential energy columns: {energy_cols}")

if energy_cols:
    target_col = energy_cols[0]
    data = daily[target_col].dropna()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histogram
    axes[0].hist(data, bins=50, color='#00d2ff', edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Distribution of {target_col}')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Frequency')

    # Box plot
    axes[1].boxplot(data, vert=True)
    axes[1].set_title(f'Box Plot — {target_col}')
    axes[1].set_ylabel(target_col)

    # QQ plot
    from scipy import stats
    stats.probplot(data.sample(min(5000, len(data)), random_state=42), plot=axes[2])
    axes[2].set_title('Q-Q Plot (Normality Check)')

    plt.tight_layout()
    plt.show()

    print(f"\nSkewness: {data.skew():.4f}")
    print(f"Kurtosis: {data.kurtosis():.4f}")
    if abs(data.skew()) > 1:
        print("Highly skewed — consider log transformation for linear models")
    else:
        print("Moderate skewness — no transformation strictly needed")

# %% [markdown]
# ### Observation
#
# The energy consumption is highly right-skewed with many low values and some very high outliers. I will have to keep this skewness in mind when choosing regression models or preprocessing.

# %% [markdown]
# ## Time-Series Trends
#
# I will look at how energy consumption changes over dates, months, and days of the week.

# %%
# Parse dates and group by day
date_candidates = [c for c in daily.columns if any(kw in c.lower() for kw in ['date', 'day', 'time'])]
print(f"Date column candidates: {date_candidates}")

if date_candidates and energy_cols:
    daily[date_candidates[0]] = pd.to_datetime(daily[date_candidates[0]], errors='coerce')

    # Group by date to get daily average across all households
    daily_agg = daily.groupby(date_candidates[0])[energy_cols[0]].mean().reset_index()
    daily_agg.columns = ['date', 'avg_consumption']
    daily_agg = daily_agg.sort_values('date')

    fig = px.line(daily_agg, x='date', y='avg_consumption',
                  title='Average Daily Consumption Across All Households',
                  labels={'date': 'Date', 'avg_consumption': 'Avg Consumption (kWh)'})
    fig.update_layout(template='plotly_dark')
    fig.show()

# %%
# Monthly average consumption
if date_candidates and energy_cols:
    daily_agg['month'] = daily_agg['date'].dt.month
    daily_agg['year'] = daily_agg['date'].dt.year
    daily_agg['month_name'] = daily_agg['date'].dt.month_name()

    monthly = daily_agg.groupby(['year', 'month', 'month_name'])['avg_consumption'].mean().reset_index()

    fig = px.bar(monthly, x='month_name', y='avg_consumption', color='year',
                 barmode='group', title='Monthly Average Consumption by Year',
                 labels={'avg_consumption': 'Avg Consumption', 'month_name': 'Month'})
    fig.update_layout(template='plotly_dark')
    fig.show()

# %%
# Day of week average consumption
if date_candidates and energy_cols:
    daily_agg['day_of_week'] = daily_agg['date'].dt.day_name()
    daily_agg['dow_num'] = daily_agg['date'].dt.dayofweek

    dow = daily_agg.groupby(['day_of_week', 'dow_num'])['avg_consumption'].mean().reset_index()
    dow = dow.sort_values('dow_num')

    fig = px.bar(dow, x='day_of_week', y='avg_consumption',
                 title='Average Consumption by Day of Week',
                 labels={'avg_consumption': 'Avg Consumption', 'day_of_week': 'Day'},
                 color='avg_consumption', color_continuous_scale='Viridis')
    fig.update_layout(template='plotly_dark')
    fig.show()

    print(f"\nHighest consumption day: {dow.loc[dow['avg_consumption'].idxmax(), 'day_of_week']}")
    print(f"   Lowest consumption day: {dow.loc[dow['avg_consumption'].idxmin(), 'day_of_week']}")

# %% [markdown]
# ### Observation
#
# Daily consumption rises during the winter months (November to February) and drops during summer, matching the seasonal temperature trends. On a weekly scale, energy consumption is slightly higher on weekends compared to weekdays.

# %% [markdown]
# ## Weather vs Consumption Correlation
#
# I will check if weather attributes correlate with daily average consumption values.

# %%
# Merge weather and daily aggregated consumption
if date_candidates and date_col and energy_cols:
    merged = daily_agg.merge(weather_daily, left_on='date', right_on=date_col[0], how='inner')
    weather_numeric = weather_daily.select_dtypes(include=[np.number]).columns.tolist()

    if weather_numeric:
        correlations = merged[['avg_consumption'] + weather_numeric].corr()['avg_consumption'].drop('avg_consumption')
        correlations = correlations.sort_values(ascending=False)

        print("Correlation with average daily energy consumption:")
        print("=" * 50)
        for feat, corr in correlations.items():
            bar = '█' * int(abs(corr) * 20)
            direction = '+' if corr > 0 else '-'
            print(f"  {feat:>35s}: {direction}{abs(corr):.4f} {bar}")

# %%
# Plot temperature vs average consumption
if date_candidates and date_col and energy_cols:
    temp_cols_merged = [c for c in merged.columns if 'temp' in c.lower() and merged[c].dtype in ['float64','int64']]
    if temp_cols_merged:
        fig = px.scatter(merged, x=temp_cols_merged[0], y='avg_consumption',
                         title=f'{temp_cols_merged[0]} vs Average Daily Consumption',
                         labels={temp_cols_merged[0]: 'Temperature', 'avg_consumption': 'Avg Consumption'},
                         trendline='lowess', opacity=0.5)
        fig.update_layout(template='plotly_dark')
        fig.show()

# %% [markdown]
# ### Observation
#
# Temperature has a negative correlation with energy consumption. This means as temperature decreases (getting colder), energy consumption increases, likely due to space heating needs.

# %% [markdown]
# ## Half-Hourly Data Exploration
#
# I will load a small sample of half-hourly consumption data to plot a heatmap of usage by hour and day of week.

# %%
# Upload sample half-hourly file
print("Upload ONE half-hourly block file (e.g., block_0.csv)")
uploaded_hh_data = files.upload()

# %%
# Load and inspect half-hourly data
hh_filename = list(uploaded_hh_data.keys())[0]
sample_hh = pd.read_csv(hh_filename)
print(f"Shape: {sample_hh.shape}")
print(f"Columns: {list(sample_hh.columns)}")
sample_hh.head()

# %%
# Generate consumption heatmap: Hour of Day vs Day of Week
dt_col_hh = [c for c in sample_hh.columns if any(kw in c.lower() for kw in ['date', 'time', 'tstp'])]
en_col_hh = [c for c in sample_hh.columns if any(kw in c.lower() for kw in ['energy', 'kwh', 'consumption'])]

if dt_col_hh and en_col_hh:
    sample_hh[dt_col_hh[0]] = pd.to_datetime(sample_hh[dt_col_hh[0]], errors='coerce')
    sample_hh[en_col_hh[0]] = pd.to_numeric(sample_hh[en_col_hh[0]], errors='coerce')
    sample_hh = sample_hh.dropna(subset=[dt_col_hh[0], en_col_hh[0]])

    sample_hh['hour'] = sample_hh[dt_col_hh[0]].dt.hour
    sample_hh['dow_num'] = sample_hh[dt_col_hh[0]].dt.dayofweek

    heatmap_data = sample_hh.groupby(['dow_num', 'hour'])[en_col_hh[0]].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='dow_num', columns='hour', values=en_col_hh[0])

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot.index = [day_names[i] for i in heatmap_pivot.index]

    plt.figure(figsize=(16, 6))
    sns.heatmap(heatmap_pivot, cmap='YlOrRd', annot=False, cbar_kws={'label': 'Avg Consumption'})
    plt.title('Energy Consumption Heatmap — Hour of Day vs Day of Week')
    plt.xlabel('Hour of Day')
    plt.ylabel('Day of Week')
    plt.tight_layout()
    plt.show()
else:
    print("Could not find datetime or energy columns automatically.")

# %% [markdown]
# ### Observation
#
# The heatmap shows two main daily peaks in consumption: a small one in the morning (around 7 AM to 9 AM) and a much larger peak in the evening (around 6 PM to 9 PM). This is common for household activity.

# %% [markdown]
# ## Missing Values & Relationships
#
# Checking missing values and weather feature correlations.

# %%
# Missing values in daily data
if not daily.empty:
    missing = daily.isnull().sum()
    missing_pct = (missing / len(daily) * 100).round(2)
    missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)

    if len(missing_df) > 0:
        print("Columns with missing values:")
        print("=" * 50)
        print(missing_df.to_string())

        fig = px.bar(missing_df, x=missing_df.index, y='Missing %',
                     title='Missing Values by Column (%)',
                     labels={'x': 'Column', 'Missing %': '% Missing'})
        fig.update_layout(template='plotly_dark')
        fig.show()
    else:
        print("No missing values found in daily dataset!")

# %%
# Correlation of weather features
weather_num = weather_daily.select_dtypes(include=[np.number])
if len(weather_num.columns) > 0:
    plt.figure(figsize=(14, 10))
    corr = weather_num.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=False, square=True,
                linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Weather Features — Correlation Matrix')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## Summary of EDA
#
# I have explored the metadata, weather, and consumption datasets. Key findings:
# - Consumption increases significantly during cold temperatures, indicating heating sensitivity.
# - High skewness is present in energy readings, indicating that some households consume much more than average.
# - Consumption peaks in the evening hours and is slightly higher on weekends.
# - Weather attributes show strong multi-collinearity, which I will need to handle in the feature engineering and modeling stages.
