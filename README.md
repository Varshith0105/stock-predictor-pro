# 📈 StockAI – AI-Powered Stock Price Prediction

StockAI is an AI-powered stock price prediction web application that fetches real-time market data and predicts future stock prices using machine learning models. The platform provides interactive dashboards, historical price analysis, and short-term future price forecasts to help users understand market trends.

---

## 🚀 Features

- 🔴 **Live Stock Market Data**
  - Fetches real-time stock prices for popular companies.

- 🤖 **AI-Based Price Prediction**
  - Uses Machine Learning models to predict future stock prices.

- 📊 **Interactive Data Visualization**
  - Displays actual vs predicted prices using dynamic charts.

- 📅 **Flexible Date Range Selection**
  - 7 Days
  - 1 Month
  - 3 Months
  - 6 Months
  - 1 Year

- ⏭️ **Future Prediction Window**
  - Predict next:
    - 3 Days
    - 7 Days
    - 14 Days
    - 30 Days

- 🔄 **Model Comparison Support**
  - Designed to support multiple prediction models.

- 🌙 **Modern Dark-Themed Dashboard**
  - Clean, responsive, and user-friendly UI.

---

## 🖥️ Application Overview

The dashboard provides:
- Current stock price
- Percentage change
- Average prediction accuracy
- Active prediction model
- Historical price trend
- Predicted future prices

Actual prices are shown as a **solid line**, while predicted prices are displayed as a **dotted line** for easy comparison.

---

## 🧠 Machine Learning Model

### 📌 Linear Regression
- Learns the relationship between time and stock price.
- Uses historical price data for training.
- Suitable for short-term forecasting.
- Fast and lightweight.

**Currently Active Model:** Linear Regression

---

## 🏗️ Project Structure

```bash
StockAI/
│
├── data/
│   ├── stock_data.csv
│
├── models/
│   ├── linear_regression.py
│
├── services/
│   ├── data_fetcher.py
│   ├── predictor.py
│
├── frontend/
│   ├── components/
│   ├── charts/
│   ├── dashboard.jsx
│
├── app.py / main.py
├── requirements.txt
└── README.md
