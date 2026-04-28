# 🏏 IPL Analytics: Real-Time Score & Win Probability Engine

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blueviolet.svg)](https://mlflow.org/)
[![Grafana](https://img.shields.io/badge/Grafana-Observability-orange.svg)](https://grafana.com/)

A comprehensive, real-time Machine Learning pipeline designed to predict outcomes for T20 (IPL) cricket matches. Powered by historical ball-by-ball data, this full-stack MLOps project provides live inference via an interactive UI, backed by a scalable microservice architecture.

## 📑 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Architecture](#-project-architecture)
- [Getting Started](#-getting-started)
- [Services & Ports](#-services--ports)
- [Project Structure](#-project-structure)

---

## 🎯 Overview
During a live IPL match, this engine answers two primary questions based on the current state of the game:
1. **1st Innings (Run Projection)**: What will be the final projected score based on current runs, wickets, and overs?
2. **2nd Innings (Run Chase)**: What is the real-time Win Probability for the chasing team given the target score?

## ✨ Features
- **Dynamic Feature Engineering**: Computes rolling statistical baselines (e.g., `career_strike_rate`, `bowling_economy`) dynamically to avoid high-cardinality issues with player names.
- **Accurate ML Models**: Utilizes `RandomForestRegressor` (~12 run RMSE) for the 1st innings and `RandomForestClassifier` (~85% Accuracy) for the 2nd innings.
- **Strict Cricket Logic Validations**: The UI enforces rules (e.g., max 6 balls per over, valid target scores constraint).
- **Automated Experiment Tracking**: Records model parameters and metrics using **MLflow**.
- **Real-Time API Observability**: Tracks inference latency (95th percentile), HTTP request duration, and API health using **Prometheus & Grafana**.
- **Data Versioning**: Manages large `.csv` dataset tracking outside of Git using **DVC**.

---

## 🛠 Tech Stack
| Component | Technology |
| --- | --- |
| **Frontend** | Streamlit |
| **Backend API** | FastAPI, Uvicorn |
| **Machine Learning** | Scikit-Learn, Pandas, Numpy, Joblib |
| **Experiment Tracking** | MLflow |
| **Data Versioning** | DVC (Data Version Control) |
| **Observability** | Prometheus, Grafana |
| **Containerization** | Docker, Docker Compose |

---

## 🚀 Getting Started

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & Docker Compose
- [Git](https://git-scm.com/)

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/humilitykind/IPL_Analytics.git
   cd IPL_Analytics
   ```

2. **Pull the Data via DVC (Optional)** 
   If you have DVC remote configured, pull the historical datasets:
   ```bash
   dvc pull
   ```
   *(Note: You can also manually place the `ipl_dataset.csv` in `data/raw/` if DVC is not configured locally).*

3. **Build and Run the Docker Containers**
   The entire microservice architecture is orchestrated via Docker Compose.
   ```bash
   docker-compose up --build -d
   ```
   This will spin up the Frontend, Backend, MLflow server, Prometheus, and Grafana containers.

---

## 🌐 Services & Ports
Once the containers are running, you can access the services at the following local ports:

| Service | URL | Description |
| --- | --- | --- |
| **Streamlit UI** | `http://localhost:8502` | The main interactive application |
| **FastAPI Backend** | `http://localhost:8000/docs` | Swagger UI for the ML REST API |
| **Grafana Dashboard** | `http://localhost:3001` | System Monitoring & Observability |
| **Prometheus** | `http://localhost:9090` | Time-series metrics scraper |
| **MLflow Tracking** | `http://localhost:5001` | ML experiment parameters and metrics |

> **Note**: Grafana visualizations have a persistent volume (`grafana_data`), so your dashboard configurations will survive container restarts.

---

## 📂 Project Structure
```text
IPL_Analytics/
├── backend/                  # FastAPI Application
│   ├── main.py               # API Endpoints (/predict_score, /predict_win)
│   ├── requirements.txt      
│   └── Dockerfile            
├── frontend/                 # Streamlit UI
│   ├── app.py                # Interactive Web Interface components
│   ├── requirements.txt      
│   └── Dockerfile            
├── src/                      # ML Source Code
│   ├── features/             # Feature Engineering Scripts (build_features.py)
│   └── models/               # Training Pipeline (train_model.py)
├── data/                     # Ignored by Git (Tracked by DVC)
│   ├── raw/                  # Raw ball-by-ball files
│   ├── processed/            # Intermediary baselines & tables
│   └── models/               # Pickled Sklearn Models (joblib)
├── prometheus/               # Prometheus configuration
│   └── prometheus.yml        
├── docker-compose.yml        # Multi-container orchestration
├── REPORT.md                 # Detailed project write-up
└── README.md                 # Project documentation
```

---

*Built with ❤️ for MLOps & Cricket Data Analytics.*
