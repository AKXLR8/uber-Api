from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import datetime
from CustomModel import WeightedEnsembleRegressor
import joblib


# Load the model

from pathlib import Path

MODEL_PATH = Path(__file__).parent / "final_uber_ensemble_model.pkl"
model = joblib.load(MODEL_PATH)


# App setup
app = Flask(__name__)
CORS(app)

# Configuration
WINDOW_SIZE = 24

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Expecting a list of 24 lag values
        lag_values = data.get("lag_values")

        if not lag_values or len(lag_values) != WINDOW_SIZE:
            return jsonify({"error": f"Please provide exactly {WINDOW_SIZE} lagged values."}), 400

        # Convert to numpy and reshape
        features = np.array(lag_values).reshape(1, -1)
        prediction = model.predict(features)[0]

        return jsonify({
            "predicted_trip_count": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Working", "model": "Uber Trip Count Ensemble Model"})

if __name__ == "__main__":
    app.run(debug=True)
