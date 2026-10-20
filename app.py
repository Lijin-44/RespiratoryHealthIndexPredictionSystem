from flask import Flask, jsonify, render_template, request
import os

import numpy as np
import pandas as pd


app = Flask(__name__)

# ==================================================
# 1. LOAD DATASET
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "respiratory_aqi_rhi_app_input.csv"
)

df = pd.read_csv(DATASET_PATH)
df.columns = df.columns.str.strip()

required_columns = [
    "city",
    "last_update",
    "Temperature_C",
    "Humidity_%",
    "Wind_Speed_kmh",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "OZONE",
    "NH3"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

numeric_columns = [
    "Temperature_C",
    "Humidity_%",
    "Wind_Speed_kmh",
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "OZONE",
    "NH3"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["last_update"] = pd.to_datetime(
    df["last_update"],
    errors="coerce"
)

cities = sorted(
    df["city"]
    .dropna()
    .astype(str)
    .str.strip()
    .loc[lambda values: values != ""]
    .unique()
    .tolist()
)

# ==================================================
# 2. DISEASE GUIDANCE
# ==================================================

# General air-quality precautions by RHI class and condition. These are
# educational prompts, not individualized medical advice.
DISEASE_RECOMMENDATIONS = {
    "asthma": {
        "name": "Asthma",
        "recommendations_by_risk": {
            "Low": [
                "Continue your existing asthma care plan and avoid your known triggers.",
                "A Low RHI does not replace monitoring your usual symptoms."
            ],
            "Moderate": [
                "Notice whether outdoor activity brings on coughing or wheezing; reduce prolonged exertion if it does.",
                "Keep following your prescribed asthma action plan."
            ],
            "High": [
                "Limit strenuous outdoor activity, especially if it worsens your asthma symptoms.",
                "Follow your prescribed asthma action plan if symptoms begin or worsen."
            ],
            "Severe": [
                "Reduce outdoor exposure where practical and follow your prescribed asthma action plan.",
                "Seek urgent medical help for severe or rapidly worsening breathing difficulty."
            ]
        }
    },
    "copd": {
        "name": "COPD",
        "recommendations_by_risk": {
            "Low": [
                "Continue your clinician-provided COPD plan and avoid smoke and other irritants.",
                "Keep track of changes from your usual breathing or symptoms."
            ],
            "Moderate": [
                "Pace outdoor activity and take breaks if you become more short of breath than usual.",
                "Avoid smoke and follow your existing COPD care plan."
            ],
            "High": [
                "Reduce strenuous outdoor exertion and avoid smoke or other respiratory irritants.",
                "Contact your healthcare professional if your usual breathing or symptoms worsen."
            ],
            "Severe": [
                "Reduce outdoor exposure where practical and follow your COPD care plan.",
                "Seek urgent medical help for severe or rapidly worsening breathlessness."
            ]
        }
    },
    "bronchitis": {
        "name": "Bronchitis",
        "recommendations_by_risk": {
            "Low": [
                "Avoid smoking and second-hand smoke while recovering.",
                "Rest and drink fluids if appropriate for you."
            ],
            "Moderate": [
                "Avoid smoke and dusty or irritating air where possible.",
                "Pace activity and watch for worsening cough or breathing symptoms."
            ],
            "High": [
                "Limit strenuous outdoor activity and avoid smoke or other irritants.",
                "Contact a healthcare professional if your cough or breathing gets worse."
            ],
            "Severe": [
                "Reduce outdoor exposure where practical and avoid smoke or other irritants.",
                "Seek prompt medical advice for worsening symptoms, and urgent help for severe breathing difficulty."
            ]
        }
    },
    "pneumonia": {
        "name": "Pneumonia",
        "recommendations_by_risk": {
            "Low": [
                "Follow the treatment and recovery advice given by your healthcare professional.",
                "Rest and maintain fluids if appropriate for you."
            ],
            "Moderate": [
                "Continue following your clinician's pneumonia care advice and avoid smoke.",
                "Contact your healthcare professional if your symptoms are not improving as expected."
            ],
            "High": [
                "Avoid strenuous outdoor activity and smoke while recovering.",
                "Contact your healthcare professional if breathing or other symptoms worsen."
            ],
            "Severe": [
                "Reduce outdoor exposure where practical and continue following your clinician's care advice.",
                "Seek urgent help for severe breathing difficulty, confusion, or blue lips."
            ]
        }
    },
    "allergic-rhinitis": {
        "name": "Allergic rhinitis",
        "recommendations_by_risk": {
            "Low": [
                "Continue avoiding allergens known to trigger your symptoms.",
                "Use only the symptom-management plan recommended for you."
            ],
            "Moderate": [
                "If air quality or outdoor allergens trigger symptoms, consider reducing time outside.",
                "Avoid rubbing irritated eyes or nose and follow your usual care plan."
            ],
            "High": [
                "Limit time outdoors if it worsens your allergy symptoms; avoid known allergens where possible.",
                "Ask a healthcare professional or pharmacist about suitable symptom relief if needed."
            ],
            "Severe": [
                "Reduce outdoor exposure where practical if it aggravates your symptoms.",
                "Seek medical advice for persistent or worsening symptoms; urgent help is warranted for severe breathing difficulty."
            ]
        }
    },
    "sinusitis": {
        "name": "Sinusitis",
        "recommendations_by_risk": {
            "Low": [
                "Avoid smoke and other irritants that make your sinus symptoms worse.",
                "Rest and drink fluids if appropriate for you."
            ],
            "Moderate": [
                "Limit exposure to irritating air if it worsens congestion or discomfort.",
                "Ask a healthcare professional about symptom relief if needed."
            ],
            "High": [
                "Reduce strenuous outdoor activity if it aggravates your symptoms and avoid smoke.",
                "Seek medical advice if symptoms are severe, persistent, or worsening."
            ],
            "Severe": [
                "Reduce outdoor exposure where practical and avoid smoke or other irritants.",
                "Contact a healthcare professional about worsening symptoms; seek urgent help for severe breathing difficulty."
            ]
        }
    },
    "influenza": {
        "name": "Influenza (flu)",
        "recommendations_by_risk": {
            "Low": [
                "Rest and drink fluids if appropriate; limit close contact with others while ill.",
                "Follow current local public-health advice."
            ],
            "Moderate": [
                "Avoid strenuous outdoor activity while unwell and limit close contact with others.",
                "Contact a healthcare professional if you are at higher risk or your symptoms worsen."
            ],
            "High": [
                "Stay home when possible while ill, limit close contact, and avoid strenuous outdoor exertion.",
                "Ask a healthcare professional for advice if symptoms are severe or you are at higher risk."
            ],
            "Severe": [
                "Reduce outdoor exposure where practical while ill and follow local public-health advice.",
                "Seek urgent help for severe breathing difficulty or sudden deterioration."
            ]
        }
    },
    "covid-19": {
        "name": "COVID-19",
        "recommendations_by_risk": {
            "Low": [
                "Follow current local public-health guidance and rest while unwell.",
                "Limit close contact with others while you may be infectious."
            ],
            "Moderate": [
                "Avoid strenuous outdoor activity while unwell and follow local public-health guidance.",
                "Contact a healthcare professional if you are at higher risk or symptoms worsen."
            ],
            "High": [
                "Stay home when possible while unwell, limit close contact, and avoid strenuous outdoor activity.",
                "Ask a healthcare professional for advice if symptoms are severe or you are at higher risk."
            ],
            "Severe": [
                "Reduce outdoor exposure where practical while unwell and follow local public-health advice.",
                "Seek urgent help for severe breathing difficulty or sudden deterioration."
            ]
        }
    }
}

# ==================================================
# 3. CALCULATE REFERENCE VALUES
# ==================================================

# Use the dataset's 95th percentile to normalize pollutant
# and wind-speed values.
pollutants = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "OZONE",
    "NH3"
]

reference_values = {}

for column in pollutants:
    values = df[column]
    values = values[np.isfinite(values) & (values >= 0)]

    reference_values[column] = (
        float(values.quantile(0.95))
        if not values.empty
        else 0.0
    )

wind_values = df["Wind_Speed_kmh"]
wind_values = wind_values[
    np.isfinite(wind_values) & (wind_values >= 0)
]

reference_values["Wind_Speed_kmh"] = (
    float(wind_values.quantile(0.95))
    if not wind_values.empty
    else 0.0
)

# ==================================================
# 4. RHI CALCULATION
# ==================================================

def calculate_rhi(
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
    def pollutant_score(value, reference):
        if reference <= 0:
            return 0.0

        return min((value / reference) * 100, 100.0)

    s_pm25 = pollutant_score(pm25, reference_values["PM2.5"])
    s_pm10 = pollutant_score(pm10, reference_values["PM10"])
    s_no2 = pollutant_score(no2, reference_values["NO2"])
    s_so2 = pollutant_score(so2, reference_values["SO2"])
    s_co = pollutant_score(co, reference_values["CO"])
    s_ozone = pollutant_score(ozone, reference_values["OZONE"])
    s_nh3 = pollutant_score(nh3, reference_values["NH3"])

    s_temperature = min(abs(temperature - 25) / 15 * 100, 100.0)
    s_humidity = min(abs(humidity - 50) / 50 * 100, 100.0)

    wind_reference = reference_values["Wind_Speed_kmh"]

    if wind_reference <= 0:
        s_wind = 0.0
    else:
        s_wind = 100 * (
            1 - min(wind_speed / wind_reference, 1.0)
        )

    rhi = (
        0.18 * s_pm25
        + 0.12 * s_pm10
        + 0.09 * s_no2
        + 0.07 * s_so2
        + 0.07 * s_co
        + 0.09 * s_ozone
        + 0.03 * s_nh3
        + 0.10 * s_temperature
        + 0.10 * s_humidity
        + 0.15 * s_wind
    )

    return round(max(0.0, min(rhi, 100.0)), 2)


def classify_rhi(rhi):
    if rhi <= 25:
        return "Low"
    if rhi <= 50:
        return "Moderate"
    if rhi <= 75:
        return "High"
    return "Severe"

# ==================================================
# 5. HOME PAGE
# ==================================================

@app.route("/")
def home():
    return render_template("index.html", cities=cities)

# ==================================================
# 6. LOAD ENVIRONMENTAL DATA FOR SELECTED CITY
# ==================================================

@app.route("/city", methods=["POST"])
def city_data():
    selected_city = request.form.get("city", "").strip()

    if selected_city not in cities:
        return "Please select a valid city.", 400

    city_records = df[
        df["city"].astype(str).str.strip() == selected_city
    ].copy()

    if city_records.empty:
        return "City not found in the dataset.", 404

    city_records = city_records.sort_values(
        "last_update",
        ascending=False,
        na_position="last"
    )

    valid_records = city_records.dropna(subset=numeric_columns)

    if valid_records.empty:
        return (
            "No complete environmental record is available for this city.",
            404
        )

    data = valid_records.iloc[0]

    return render_template(
        "index.html",
        cities=cities,
        selected_city=selected_city,
        temperature=float(data["Temperature_C"]),
        humidity=float(data["Humidity_%"]),
        wind_speed=float(data["Wind_Speed_kmh"]),
        pm25=float(data["PM2.5"]),
        pm10=float(data["PM10"]),
        no2=float(data["NO2"]),
        so2=float(data["SO2"]),
        co=float(data["CO"]),
        ozone=float(data["OZONE"]),
        nh3=float(data["NH3"])
    )

# ==================================================
# 7. CALCULATE RHI AND DISPLAY RESULT
# ==================================================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        selected_city = request.form.get("city", "").strip()

        if not selected_city:
            return "Please select a city.", 400

        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        wind_speed = float(request.form["wind_speed"])
        pm25 = float(request.form["pm25"])
        pm10 = float(request.form["pm10"])
        no2 = float(request.form["no2"])
        so2 = float(request.form["so2"])
        co = float(request.form["co"])
        ozone = float(request.form["ozone"])
        nh3 = float(request.form["nh3"])

        inputs = [
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
        ]

        if not all(np.isfinite(value) for value in inputs):
            return "Please enter valid numerical values.", 400

        if not 0 <= humidity <= 100:
            return "Humidity must be between 0 and 100.", 400

        pollutant_inputs = [pm25, pm10, no2, so2, co, ozone, nh3]

        if wind_speed < 0 or any(value < 0 for value in pollutant_inputs):
            return (
                "Wind speed and pollutant values cannot be negative.",
                400
            )

        rhi = calculate_rhi(
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            pm25=pm25,
            pm10=pm10,
            no2=no2,
            so2=so2,
            co=co,
            ozone=ozone,
            nh3=nh3
        )

        risk_level = classify_rhi(rhi)

        disease_guidance = [
            {
                "id": disease_id,
                "name": disease["name"],
                "recommendations": disease["recommendations_by_risk"][risk_level]
            }
            for disease_id, disease in DISEASE_RECOMMENDATIONS.items()
        ]

        return render_template(
            "result.html",
            selected_city=selected_city,
            rhi=rhi,
            predicted_risk=risk_level,
            diseases=disease_guidance,
            temperature=temperature,
            humidity=humidity,
            wind_speed=wind_speed,
            pm25=pm25,
            pm10=pm10,
            no2=no2,
            so2=so2,
            co=co,
            ozone=ozone,
            nh3=nh3
        )

    except (ValueError, TypeError, KeyError):
        return (
            "Invalid or missing input. Please enter all ten environmental values.",
            400
        )

# ==================================================
# 8. DISEASE RECOMMENDATION API
# ==================================================

@app.route("/api/recommendations/<disease_id>")
def disease_recommendation(disease_id):
    disease = DISEASE_RECOMMENDATIONS.get(disease_id)

    if disease is None:
        return jsonify({"error": "Disease not found."}), 404

    risk_level = request.args.get("risk_class", "").strip().title()
    if risk_level not in {"Low", "Moderate", "High", "Severe"}:
        return jsonify({
            "error": "Provide a valid risk_class: Low, Moderate, High, or Severe."
        }), 400

    return jsonify({
        "id": disease_id,
        "name": disease["name"],
        "risk_class": risk_level,
        "recommendations": disease["recommendations_by_risk"][risk_level]
    })

# ==================================================
# 9. RUN APPLICATION
# ==================================================

if __name__ == "__main__":
    app.run(debug=True)