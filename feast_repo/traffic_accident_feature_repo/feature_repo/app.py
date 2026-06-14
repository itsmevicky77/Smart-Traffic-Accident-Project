from fastapi import FastAPI
import pandas as pd
import joblib
import os

# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Traffic Accident Prediction API",
    version="1.0"
)

# =====================================================
# LOAD MODEL
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../../../models/best_model.pkl"
    )
)

model = joblib.load(MODEL_PATH)

# =====================================================
# HOME ROUTE
# =====================================================

@app.get("/")
def home():

    return {
        "message": "Traffic Accident Prediction API Running Successfully"
    }

# =====================================================
# PREDICTION ROUTE
# =====================================================

@app.post("/predict")
def predict(
    lane_count: int,
    speed_limit_kmph: int,
    blackspot_score: float,
    vehicle_count_per_hr: float,
    avg_speed_kmph: float,
    green_duration_s: float,
    red_duration_s: float,
    yellow_duration_s: float,
    cycle_time_s: float,
    violations_count: float,
    speed_ratio: float,
    traffic_density: float,
    signal_efficiency: float,
    is_weekend: int,
    is_night: int
):

    data = pd.DataFrame([{
        "lane_count": lane_count,
        "speed_limit_kmph": speed_limit_kmph,
        "blackspot_score": blackspot_score,
        "vehicle_count_per_hr": vehicle_count_per_hr,
        "avg_speed_kmph": avg_speed_kmph,
        "green_duration_s": green_duration_s,
        "red_duration_s": red_duration_s,
        "yellow_duration_s": yellow_duration_s,
        "cycle_time_s": cycle_time_s,
        "violations_count": violations_count,
        "speed_ratio": speed_ratio,
        "traffic_density": traffic_density,
        "signal_efficiency": signal_efficiency,
        "is_weekend": is_weekend,
        "is_night": is_night
    }])

    prediction = model.predict(data)[0]

    probability = None

    try:
        probability = float(
            model.predict_proba(data)[0][1]
        )
    except:
        pass

    return {
        "accident_predicted": bool(prediction),
        "accident_probability": probability
    }

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }