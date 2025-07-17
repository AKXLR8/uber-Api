from flask import Flask, request, jsonify
import joblib
import numpy as np

# ✅ Load the compressed model
MODEL_PATH = "final_uber_ensemble_model_compressed.pkl"
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

@app.route('/')
def home():
    return "Uber Trip Prediction API (Compressed Model)"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        lag_values = data['lag_values']

        if len(lag_values) != 24:
            return jsonify({"error": "Exactly 24 lag values are required."}), 400

        features = np.array([lag_values])
        prediction = model.predict(features)[0]

        return jsonify({
            "input_lags": lag_values,
            "predicted_trips": int(round(prediction))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
