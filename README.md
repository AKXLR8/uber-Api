
# Uber Trip Demand Forecaster

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-API-green?logo=flask)](https://flask.palletsprojects.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange?logo=xgboost)](https://xgboost.ai/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-yellow?logo=scikitlearn)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
[![Cloud Run](https://img.shields.io/badge/Google%20Cloud%20Run-Deployed-lightgrey?logo=googlecloud)](https://cloud.google.com/run)

> **Forecast Uber trip demand using ensemble machine learning models (XGBoost, Random Forest, GBRT)**
> Deployed with **Flask** and **Docker**, ready for scalable use on **Google Cloud Run**.

---

## 🧭 Overview

The **Uber Trip Demand Forecaster** is a machine learning system designed to predict **hourly trip demand** based on historical Uber data.
By combining multiple regression algorithms through an **ensemble learning framework**, it produces robust, high-accuracy demand forecasts that can inform fleet optimization and pricing strategies.

---

## 🧠 Features

* 📈 **Time Series Forecasting** using lag-based feature engineering
* 🧩 **Weighted Ensemble Model** combining XGBoost, Random Forest & GBRT
* 🔬 **Exploratory Data Analysis (EDA)** with trend and seasonality decomposition
* ⚙️ **REST API** built in Flask for real-time inference
* 🐳 **Dockerized Deployment** compatible with Google Cloud Run
* 📊 **Visualization Support** for model evaluation

---

## 🗂️ Project Structure

```
Uber-Trip-Forecaster/
├── uber_trip.py                  # Core ML pipeline (training & evaluation)
├── UberFlask.py                  # Flask API for predictions
├── final_uber_ensemble_model.pkl # Trained ensemble model
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Containerization setup
└── README.md                     # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/uber-trip-forecaster.git
cd uber-trip-forecaster
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 4️⃣ Run Flask API

```bash
python UberFlask.py
```

Your local API will start at **[http://localhost:8080](http://localhost:8080)**

---

## 🧩 API Endpoints

### ▶️ **POST** `/predict`

Predict the next hour’s Uber trip count based on 24 previous hourly values.

#### 🔹 Example Request

```json
{
  "lag_values": [152, 148, 160, 158, 172, 190, 201, 210, 205, 193, 182, 175,
                 168, 160, 154, 149, 155, 167, 180, 190, 195, 188, 172, 160]
}
```

#### 🔹 Example Response

```json
{
  "predicted_trip_count": 184.73
}
```

---

### ✅ **GET** `/health`

Check API and model status.

**Response**

```json
{"status": "Working", "model": "Uber Trip Count Ensemble Model"}
```

---

## 🧮 Model Workflow

1. **Data Loading & Cleaning**

   * Combines Uber 2014 CSV datasets
   * Converts timestamps to hourly aggregates

2. **Feature Engineering**

   * Extracts `Hour`, `Day`, `Month`, `DayOfWeek`
   * Builds 24-hour lag features for time dependency

3. **Model Training**

   * Trains **Random Forest**, **GBRT**, and **XGBoost** models
   * Evaluates using **MAPE**, **R²**, and visual plots

4. **Ensemble Prediction**

   * Combines predictions via a **WeightedEnsembleRegressor**
   * Saves final model as `final_uber_ensemble_model.pkl`

5. **Deployment**

   * Flask API loads ensemble model
   * Docker container runs on `0.0.0.0:8080` for Cloud Run

---

## 🧾 Requirements

From [`requirements.txt`](requirements.txt):

```
flask
flask-cors
joblib
numpy
scikit-learn
xgboost
gunicorn
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t uber-trip-forecaster .
```

### Run Container Locally

```bash
docker run -p 8080:8080 uber-trip-forecaster
```

---

## ☁️ Google Cloud Run Deployment

1. **Build & Push Image**

   ```bash
   gcloud builds submit --tag gcr.io/<PROJECT-ID>/uber-trip-forecaster
   ```
2. **Deploy to Cloud Run**

   ```bash
   gcloud run deploy uber-trip-forecaster \
     --image gcr.io/<PROJECT-ID>/uber-trip-forecaster \
     --platform managed \
     --region asia-south1 \
     --allow-unauthenticated
   ```
3. Access your API at the deployed URL (e.g.,
   `https://uber-trip-forecaster-<hash>.a.run.app/predict`)

---

## 📊 Visual Insights

Below are sample output visuals from `uber_trip.py` (you can embed your actual charts later):

| Visualization                  | Description                              |
| ------------------------------ | ---------------------------------------- |
| 🕐 **Hourly Trend Plot**       | Trip volume variation over time          |
| 🌤️ **Seasonal Decomposition** | Seasonal, trend, and residual components |
| 🌲 **Feature Importance**      | Contribution of hour, day, and month     |
| 📈 **Model Comparison**        | Ensemble vs individual model predictions |

*(Add your screenshots in `/assets` and embed them here using Markdown)*

---

## 🧩 Sample Chart Embeds

```markdown
![Trip Trend](assets/trip_trend.png)
![Model Comparison](assets/model_comparison.png)
```

---

## 🧭 Future Enhancements

* 🔁 Automate retraining pipeline (Airflow or Kubeflow)
* 💾 Integrate **MLflow** or **Vertex AI** for experiment tracking
* 📈 Add LSTM / TFT models for deep temporal learning
* 🌐 Build Streamlit dashboard for real-time visualization

---
