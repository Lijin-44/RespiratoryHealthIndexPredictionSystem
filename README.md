# Respiratory Health Index Prediction System

A Flask web application that lets users review environmental readings for a
city in the bundled dataset and calculate an estimated Respiratory Health
Index (RHI). The result page also provides general air-quality precautions.

The web application calculates RHI using the formula in `app.py`. Separately,
`predict.py` loads the saved XGBoost classifier in `model/` and demonstrates a
respiratory-risk class prediction from manually supplied readings.

> **Health notice:** RHI and the recommendations are for general awareness
> only. They are not a medical diagnosis or a substitute for professional
> medical advice.

## Requirements

- Python 3.10 or newer
- The packages listed in `requirements.txt`

## Run locally

From the project directory, create and activate a virtual environment, then
install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the Flask application:

```powershell
python app.py
```

Open <http://127.0.0.1:5000> in a browser. The Flask development server is
intended for local development, not production deployment.

To run the standalone saved-model example:

```powershell
python predict.py
```

## Project structure

```text
app.py                 Flask application and RHI calculation
predict.py             Standalone saved-model prediction example
dataset/               Environmental readings used by the application
model/                 Serialized XGBoost model and metadata
static/css/            Application styles
templates/             Flask HTML templates
requirements.txt       Python dependencies
```

The model files use Python serialization. Only load model artifacts from a
source you trust. Confirm the dataset's source and redistribution terms before
reusing it outside this project.
