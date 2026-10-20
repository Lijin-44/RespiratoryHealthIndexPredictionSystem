import os
import joblib
import pandas as pd


# Get the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# Model file paths
MODEL_PATH = os.path.join(BASE_DIR, "model", "xgboost_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "features.pkl")


# Load trained model
model = joblib.load(MODEL_PATH)

# Load label encoder
label_encoder = joblib.load(ENCODER_PATH)

# Load feature names
features = joblib.load(FEATURES_PATH)


def predict_respiratory_risk(
    temperature,
    humidity,
    wind_speed,
    pm25,
    pm10,
    no2,
    so2,
    co,
    ozone,
    nh3
):

    # Create input data
    input_data = {
        "Temperature_C": temperature,
        "Humidity_%": humidity,
        "Wind_Speed_kmh": wind_speed,
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "CO": co,
        "OZONE": ozone,
        "NH3": nh3
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Arrange columns in the same order used during training
    input_df = input_df[features]

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Convert encoded prediction to class name
    predicted_class = label_encoder.inverse_transform([prediction])[0]

    return predicted_class


# Test the model
if __name__ == "__main__":

    result = predict_respiratory_risk(
        temperature=23,
        humidity=67,
        wind_speed=14,
        pm25=45,
        pm10=78,
        no2=30,
        so2=16,
        co=0.8,
        ozone=24,
        nh3=5
    )

    print("Predicted Respiratory Risk:", result)