# 🌐 Atmospheric Intelligence & AQI Prediction Platform

An end-to-end production-grade MLOps ecosystem delivering real-time Air Quality Index (AQI) tracking, 72-hour forecasting, and WebGL-powered 3D atmospheric simulations.

🔗 **Live Application:** [Executive Atmospheric Hub](https://wheatherpredaction-shfuvmk8sb7kglpr4hksdh.streamlit.app)

---

## 🏗️ Architecture Overview

* **Feature Store & Model Registry:** Hopsworks Cloud (`eu-west.cloud.hopsworks.ai`) managing online/offline feature pipelines and model versioning.
* **Data Sources:** Real-time continuous ingestion from **AQICN (World Air Quality Index)** and **OpenWeatherMap APIs**.
* **Automated MLOps Pipeline:** GitHub Actions running automated hourly cron jobs for feature extraction, engineering, and ingestion into Hopsworks.
* **Machine Learning Engine:** Baseline Ridge/Tree models and deep sequence architectures evaluated with dynamic **SHAP (SHapley Additive exPlanations)** causal inference.
* **Executive Frontend:** Streamlit with embedded **Three.js WebGL 3D particle sphere**, responsive Dark Glassmorphism UI, and Plotly predictive waveforms.

---

## ✨ Key Features

* **Global City Geocoding & Search:** Instant multi-city lookup with resilient fallback and quick preset navigation (London, Peshawar, Lahore, Islamabad, Karachi, New York, Tokyo).
* **Interactive 3D Atmospheric Particle Mesh:** Real-time WebGL rendering reacting dynamically to live air quality severity.
* **72-Hour Predictive Waveform:** Interactive forecasts with 95% confidence bands and EPA category health advisories.
* **Explainable AI (XAI):** Real-time SHAP feature contribution bar charts explaining meteorological impacts on AQI predictions.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10
* **Frameworks & Libraries:** Streamlit, Three.js, Plotly, Scikit-Learn, SHAP, Pandas, NumPy
* **Cloud & Infrastructure:** Hopsworks Feature Store, GitHub Actions (CI/CD), Streamlit Community Cloud

---

## ⚙️ Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Hasnain-khan10/wheather_predaction.git](https://github.com/Hasnain-khan10/wheather_predaction.git)
   cd wheather_predaction

   Create and activate a virtual environment:

Bash
python -m venv .venv
source .venv/bin/activate   # Linux / Mac
.venv\Scripts\activate      # Windows
Install dependencies:

Bash
pip install -r requirements.txt
Set up environment variables:
Create a .env file in the root directory:

Code snippet
AQICN_API_KEY=your_aqicn_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
HOPSWORKS_API_KEY=your_hopsworks_api_key
HOPSWORKS_PROJECT_NAME=shine
HOPSWORKS_HOST=eu-west.cloud.hopsworks.ai
Run the application:

Bash
python -m streamlit run frontend/app.py
