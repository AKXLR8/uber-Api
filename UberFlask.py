from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

# =========================
# Load ML Model
# =========================
MODEL_PATH = "final_uber_ensemble_model_compressed.pkl"
model = joblib.load(MODEL_PATH)

# =========================
# Initialize Flask App
# =========================
app = Flask(__name__)

# Enable CORS for frontend requests
CORS(app)

# =========================
# Home Route
# =========================
@app.route('/')
def home():
    return jsonify({
        "message": "Uber Trip Prediction API",
        "status": "running"
    })

# =========================
# Prediction Route
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Validate JSON
        if not data:
            return jsonify({
                "success": False,
                "error": "Invalid or missing JSON body"
            }), 400

        # Get lag values
        lag_values = data.get('lag_values')

        # Validate lag_values existence
        if lag_values is None:
            return jsonify({
                "success": False,
                "error": "lag_values field is required"
            }), 400

        # Validate length
        if len(lag_values) != 24:
            return jsonify({
                "success": False,
                "error": "Exactly 24 lag values are required"
            }), 400

        # Convert input to numpy array
        features = np.array([lag_values])

        # Predict
        prediction = model.predict(features)[0]

        # Return response
        return jsonify({
            "success": True,
            "input_lags": lag_values,
            "predicted_trips": int(round(prediction))
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# =========================
# Run Flask App
# =========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host='0.0.0.0',
        port=port,
        debug=True
    )
