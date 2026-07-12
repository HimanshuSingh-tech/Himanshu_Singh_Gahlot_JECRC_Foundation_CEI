import os
import sys
import joblib
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Initialize FastAPI app
app = FastAPI(
    title="EnergyLens-AI API Server",
    description="Provides predictions and profiles for the smart meter dashboard",
    version="1.0.0"
)

# Enable CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PROFILES_PATH = os.path.join(BASE_DIR, "data", "processed", "household_profiles.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "master_features.parquet")

# Recommendation Engine helper
def get_recommendations(cluster_id):
    recommendations = {
        0: [
            "Segment Archetype: Thermal Sensitive",
            "Observation: Daily energy consumption spikes dramatically during cold temperatures, indicating high dependency on electric space heating.",
            "Conservation Strategy: Focus on structural thermal efficiency. Inspect insulation, loft seals, and window glazing. Installing a smart thermostat can reduce heating costs by up to 15%.",
            "Action Step: Adjust thermostatic radiator valves (TRVs) to heat active living spaces only."
        ],
        1: [
            "Segment Archetype: Frugal & Efficient",
            "Observation: Baseline daily energy usage is consistently low, showing optimal base load management.",
            "Tariff Strategy: Leverage Time-of-Use (ToU) tariffs. Shift high-load operations (laundry, dishwashing) to off-peak slots to take advantage of cheaper energy rates.",
            "Action Step: Configure high-power smart appliances to run during off-peak slots (typically overnight or mid-afternoon)."
        ],
        2: [
            "Segment Archetype: Weekend Centric",
            "Observation: Consumption is significantly higher on weekends compared to weekdays, indicating peak usage shifting capacity.",
            "Shifting Strategy: Maximize savings by scheduling heavy tasks (EV charging, laundry, water heating) to weekend hours, matching weekend-incentive tariff rates.",
            "Action Step: Program appliances and chargers to activate during off-peak weekend hours."
        ]
    }
    return recommendations.get(cluster_id, ["No recommendations available."])

# Cache loaded models and profiles
class ModelLoader:
    def __init__(self):
        self.forecaster = None
        self.scaler = None
        self.clusterer = None
        self.profiles_df = None
        self.profiles_dict = {}
        
    def load(self):
        if self.forecaster is None:
            self.forecaster = joblib.load(os.path.join(MODELS_DIR, 'xgboost_forecaster.joblib'))
            self.scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
            self.clusterer = joblib.load(os.path.join(MODELS_DIR, 'kmeans_clusterer.joblib'))
            
            if os.path.exists(PROFILES_PATH):
                self.profiles_df = pd.read_csv(PROFILES_PATH)
                self.profiles_dict = self.profiles_df.set_index('LCLid').to_dict(orient='index')
            else:
                raise FileNotFoundError(f"Missing household profiles: {PROFILES_PATH}")

loader = ModelLoader()

# Pydantic Schemas
class WeatherDayInput(BaseModel):
    date: str
    temp_avg: float
    HDD: float
    CDD: float
    temp_range: float
    is_weekend: int
    is_holiday: int

class ForecastRequest(BaseModel):
    lclid: str
    weather: List[WeatherDayInput]

# Endpoints
@app.on_event("startup")
def startup_event():
    try:
        loader.load()
        print("[SUCCESS] Models and profiles loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Error during startup: {e}")

@app.get("/api/households")
def get_households():
    try:
        loader.load()
        households = sorted(loader.profiles_df['LCLid'].tolist())
        return {"households": households}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/profile/{lclid}")
def get_profile(lclid: str):
    try:
        loader.load()
        if lclid not in loader.profiles_dict:
            raise HTTPException(status_code=404, detail="Household profile not found")
        
        profile = loader.profiles_dict[lclid]
        recs = get_recommendations(int(profile['cluster']))
        
        # Estimate anomaly threshold: mean + 2 * std
        anomaly_threshold = float(profile['mean_daily_consumption'] + 2 * profile['std_consumption'])
        
        acorn_groups = {
            1: "Affluent (ACORN-A)", 2: "Affluent (ACORN-B)", 3: "Affluent (ACORN-C)",
            4: "Comfortable (ACORN-D)", 5: "Comfortable (ACORN-E)", 6: "Comfortable (ACORN-F)",
            7: "Comfortable (ACORN-G)", 8: "Comfortable (ACORN-H)", 9: "Comfortable (ACORN-I)",
            10: "Comfortable (ACORN-J)", 11: "Adversity (ACORN-K)", 12: "Adversity (ACORN-L)",
            13: "Adversity (ACORN-M)", 14: "Adversity (ACORN-N)", 15: "Adversity (ACORN-O)",
            16: "Adversity (ACORN-P)", 17: "Adversity (ACORN-Q)"
        }
        acorn_code = int(profile.get('acorn_code', 5))
        
        return {
            "lclid": lclid,
            "cluster_id": int(profile['cluster']),
            "cluster_archetype": recs[0].split(': ')[1],
            "mean_daily_consumption": float(profile['mean_daily_consumption']),
            "std_consumption": float(profile['std_consumption']),
            "thermal_sensitivity": float(profile['thermal_sensitivity']),
            "weekend_bias": float(profile['weekend_bias']),
            "acorn_group": acorn_groups.get(acorn_code, "Comfortable (ACORN-E)"),
            "anomaly_threshold": anomaly_threshold,
            "recommendations": recs[1:]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history/{lclid}")
def get_history(lclid: str):
    try:
        if not os.path.exists(FEATURES_PATH):
            raise HTTPException(status_code=500, detail="Feature matrix not found")
        
        # Read from parquet filtered by LCLid
        table = pq.read_table(FEATURES_PATH, filters=[('LCLid', '==', lclid)])
        df = table.to_pandas()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No historical records found")
        
        df = df.sort_values('day').reset_index(drop=True)
        
        # Convert date to string
        df['day_str'] = df['day'].astype(str)
        
        # Calculate recent 30 days
        recent = df.tail(30)
        
        # For scatter plot: select all records but drop NaNs in temp/energy
        clean_df = df.dropna(subset=['temp_avg', 'energy_mean'])
        
        # Compute regression parameters for JS
        if len(clean_df) > 5:
            slope, intercept = np.polyfit(clean_df['temp_avg'], clean_df['energy_mean'], 1)
        else:
            slope, intercept = 0.0, 0.0
            
        # Calculate weekday vs weekend averages
        we_mean = float(df[df['is_weekend'] == 1]['energy_mean'].mean()) if not df[df['is_weekend'] == 1].empty else 0.0
        wd_mean = float(df[df['is_weekend'] == 0]['energy_mean'].mean()) if not df[df['is_weekend'] == 0].empty else 0.0
            
        return {
            "total_records": len(df),
            "recent_30_days": {
                "dates": recent['day_str'].tolist(),
                "values": recent['energy_mean'].tolist(),
                "temp": recent['temp_avg'].tolist()
            },
            "scatter_data": {
                "temp": clean_df['temp_avg'].tolist(),
                "energy": clean_df['energy_mean'].tolist(),
                "is_weekend": clean_df['is_weekend'].tolist()
            },
            "regression": {
                "slope": float(slope),
                "intercept": float(intercept)
            },
            "day_type_averages": {
                "weekday": wd_mean,
                "weekend": we_mean
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forecast")
def post_forecast(req: ForecastRequest):
    try:
        loader.load()
        if req.lclid not in loader.profiles_dict:
            raise HTTPException(status_code=404, detail="Household profile not found")
            
        # 1. Fetch profile
        profile = loader.profiles_dict[req.lclid]
        
        # 2. Get recent history from parquet to calculate lags
        table = pq.read_table(FEATURES_PATH, filters=[('LCLid', '==', req.lclid)])
        hist_df = table.to_pandas()
        if hist_df.empty:
            raise HTTPException(status_code=404, detail="No historical records found for lag construction")
            
        hist_df = hist_df.sort_values('day').reset_index(drop=True)
        
        # Lags calculations from the last available actual records
        last_val = float(hist_df['energy_mean'].iloc[-1])
        last_mean_7 = float(hist_df['energy_mean'].iloc[-7:].mean())
        last_std_7 = float(hist_df['energy_mean'].iloc[-7:].std())
        last_mean_30 = float(hist_df['energy_mean'].mean())
        
        lag_1 = last_val
        lag_7 = float(hist_df['energy_mean'].iloc[-7]) if len(hist_df) >= 7 else last_val
        lag_14 = float(hist_df['energy_mean'].iloc[-14]) if len(hist_df) >= 14 else last_val
        lag_28 = float(hist_df['energy_mean'].iloc[-28]) if len(hist_df) >= 28 else last_val
        
        # 3. Build prediction matrix
        records = []
        for w in req.weather:
            records.append({
                'energy_lag_1': lag_1,
                'energy_lag_7': lag_7,
                'energy_lag_14': lag_14,
                'energy_lag_28': lag_28,
                'energy_roll_mean_7': last_mean_7,
                'energy_roll_std_7': last_std_7,
                'energy_roll_mean_30': last_mean_30,
                'temp_avg': w.temp_avg,
                'HDD': w.HDD,
                'CDD': w.CDD,
                'temp_range': w.temp_range,
                'is_weekend': w.is_weekend,
                'is_holiday': w.is_holiday,
                'tariff_code': profile.get('tariff_code', 0),
                'acorn_code': profile.get('acorn_code', 5)
            })
            
        inf_df = pd.DataFrame(records)
        
        features_order = [
            'energy_lag_1', 'energy_lag_7', 'energy_lag_14', 'energy_lag_28',
            'energy_roll_mean_7', 'energy_roll_std_7', 'energy_roll_mean_30',
            'temp_avg', 'HDD', 'CDD', 'temp_range', 'is_weekend', 'is_holiday',
            'tariff_code', 'acorn_code'
        ]
        
        X_inf = inf_df[features_order]
        
        # Run Model Prediction
        preds = loader.forecaster.predict(X_inf)
        
        # Anomaly Detection
        anomaly_threshold = float(profile['mean_daily_consumption'] + 2 * profile['std_consumption'])
        anomalous_days = []
        for i, val in enumerate(preds):
            if val > anomaly_threshold:
                anomalous_days.append(req.weather[i].date)
                
        return {
            "forecast": [float(x) for x in preds.round(4)],
            "anomaly_threshold": anomaly_threshold,
            "anomalous_days_detected": anomalous_days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static folder
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    # Make sure static directory exists
    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=8001)
