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
```
## 📉 Supported Stocks

- Apple (AAPL)
- Amazon (AMZN)
- Tesla (TSLA)
- Microsoft (MSFT)
- Meta (META)
- Netflix (NFLX)
- IBM

_(More stocks can be added easily)_

---

## 🧪 Technologies Used

### Frontend
- React.js / Streamlit
- Chart.js / Recharts
- Tailwind CSS
- Dark UI Theme

### Backend
- Python
- Pandas
- NumPy
- Scikit-learn
- Stock Market APIs

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/stockai.git
cd stockai
```
### 2️⃣ Create Virtual Environment (Optional)
```
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```
###  3️⃣ Install Dependencies
```
pip install -r requirements.txt

```
### 4️⃣ Run the Application
```
python app.py
```
## 📊 How Prediction Works

1. Fetch historical stock price data  
2. Preprocess and clean the data  
3. Train the Linear Regression model  
4. Predict future prices based on the selected date range  
5. Visualize actual vs predicted prices  

---

## 🔮 Future Enhancements

- 📈 LSTM & ARIMA models  
- 📊 Model accuracy comparison dashboard  
- 🔐 User authentication  
- ☁️ Cloud deployment  
- 📱 Mobile responsiveness  
- 📰 News-based sentiment analysis  

---

## ⚠️ Disclaimer

This project is developed **for educational and learning purposes only**.  
It is **not financial advice**. Stock market investments involve risk.  
Always consult a financial advisor before making investment decisions.

---

## 👨‍💻 Author

**Julakanti Varshith** \
**Likita Reddy Bojja**\
**Yashashwitha Kuruva**\
(AI & Machine Learning students)

---

⭐ If you like this project, consider giving it a star on GitHub!
