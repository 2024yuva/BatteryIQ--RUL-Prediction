# BatteryIQ--RUL-Prediction

BatteryIQ is an end-to-end machine learning system that predicts the **Remaining Useful Life (RUL)** of lithium-ion batteries using deep learning and Incremental Capacity (IC) analysis.
It enables early detection of degradation and supports predictive maintenance in energy systems.

---
## Built for Girls in Code — 2023 AI Challenge  

As part of the Girls in Code 2023 AI Challenge, we built BatteryIQ to address real-world challenges in battery degradation and energy efficiency.  
The project reflects our goal of using AI for sustainable innovation and smarter energy systems.

---

### Team: BatteryIQ

| Name            | GitHub ID  | 
|-----------------|------------|
| Yuvarrunjitha   | @2024yuva  |
| Hemalatha S     | @hema027   |

> Built collaboratively with a focus on real-world impact and sustainable AI.

---
## Problem Statement

Battery degradation leads to unexpected failures, safety risks, and increased costs in electric vehicles and energy storage systems.
Traditional methods fail to provide accurate early predictions.

---

## Solution

BatteryIQ predicts battery life using:

* Time-series sensor data
* IC curve feature extraction (dQ/dV)
* Bidirectional LSTM deep learning model

---

## System Architecture

<img width="1001" height="1024" alt="image" src="https://github.com/user-attachments/assets/1e2d4634-30d1-410a-bfe0-60f16f562b4d" />

This is the complete end-to-end system architecture for your Battery RUL prediction project. It shows 6 layers:

*Layer 1* — Data source: 4 NASA Li-ion cells (B0005–B0018), each recording voltage, current, temperature and capacity per cycle.
*Layer 2* — Preprocessing: Three parallel steps — IC curve computation (dQ/dV), RUL label generation at the 70% EOL threshold, and MinMaxScaler normalisation with a 10-cycle sliding window.
*Layer 3* — Feature vector: 11 features per cycle grouped into electrical (V, I), thermal (temp), capacity, and IC features. The IC features (highlighted in coral) are the key differentiator — they give early warning 20–30 cycles before visible degradation.
*Layer 4* — Bidirectional LSTM: Input → BiLSTM(64) → LSTM(32) → Dense(16) → RUL output. Trained on cells B0005–B0007, tested on B0018.
*Layer 5* — Evaluation: RMSE 2.26 cycles, MAE 1.75, ~2% relative error, best at epoch 28.
*Layer 6* — Deployment: The HTML UI talks to the Colab Flask backend over REST/JSON via your ngrok or colab.dev tunnel.

---

## Tech Stack

### Frontend

* HTML5 + CSS3
* Vanilla JavaScript
* Custom SVG-based charts
* Browser APIs (FileReader for file upload)
* Google Fonts (CDN)

---

### Machine Learning

* TensorFlow / Keras (BiLSTM model)
* NumPy (numerical operations)
* Pandas (data processing)
* Scikit-learn (MinMaxScaler, preprocessing)
* SciPy (IC curve computation, smoothing, interpolation)
* Matplotlib (visualization)

---

### Backend

* No traditional backend framework implemented
* Model runs via Python scripts and Google Colab Notebook

---

### Communication Layer

* Static frontend (no API integration currently)
* Future scope: Flask/FastAPI for real-time prediction

---

## Key Features

* Early degradation detection (20–30 cycles earlier)
* High accuracy (~2–5% error)
* IC-based feature engineering (research-level)
* Clean interactive UI
* End-to-end ML pipeline

---

## Workflow

1. Load battery dataset
2. Preprocess data and extract IC features
3. Generate feature sequences (sliding window)
4. Predict RUL using BiLSTM model
5. Visualize results

---

## Results

* RMSE: 2.26 cycles
* MAE: 1.75 cycles
* ~2–5% relative error

---

## Future Work

* Real-time API integration (Flask/FastAPI)
* IoT-based live monitoring
* Explainable AI features
* Multi-battery generalization

---

## 🙌 Acknowledgment

NASA Prognostics Center of Excellence Dataset

---

## Conclusion

BatteryIQ demonstrates how combining deep learning with Incremental Capacity analysis enables accurate and early prediction of battery degradation, contributing to sustainable and efficient energy systems.
