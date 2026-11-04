# House-price-prediction
# 🏡 House Price Prediction Studio

An interactive Machine Learning and Data Science application that predicts residential property prices using property characteristics and provides insights into housing market trends through a professional Streamlit dashboard.

## 📌 Project Overview

House Price Prediction Studio is an end-to-end Data Science project designed to analyze housing data, identify property price patterns, and predict estimated house prices using Machine Learning algorithms.

The application combines data preprocessing, exploratory data analysis, machine learning, model evaluation, and interactive visualizations in one dashboard.

## 🎯 Objectives

- Analyze housing market trends and property characteristics.
- Understand the relationship between property features and prices.
- Build and compare multiple Machine Learning regression models.
- Predict house prices based on user-provided property details.
- Present insights through interactive dashboards and visualizations.

## ✨ Key Features

- **Market Overview:** Explore housing prices, distributions, and market trends.
- **Price Prediction:** Predict estimated property prices using trained ML models.
- **Location Intelligence:** Compare property prices across different cities.
- **Model Performance:** Evaluate and compare regression algorithms.
- **Property Explorer:** Search, filter, and explore individual property records.
- **Interactive Visualizations:** Explore insights using Plotly charts.
- **Data Export:** Download filtered property datasets.

## 🧠 Machine Learning Algorithms

The project uses the following regression algorithms:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

Models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Plotly |
| Machine Learning | Scikit-learn |
| Dashboard | Streamlit |
| Model Storage | Joblib |
| Development Environment | VS Code |

## 📂 Project Structure

```text
House_Price_Prediction_Studio/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── data.csv
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── evaluation.py
│
├── pages/
│   ├── 1_Market_Overview.py
│   ├── 2_Price_Prediction.py
│   ├── 3_Location_Intelligence.py
│   ├── 4_Model_Performance.py
│   └── 5_Property_Explorer.py
│
├── models/
│   └── best_model.joblib
│
└── assets/
    └── style.css
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd House_Price_Prediction_Studio
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning models

```bash
python -m src.train_model
```

### 5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 📊 Dataset Description

The project uses a housing dataset containing property-related information, including:

- Number of bedrooms and bathrooms
- Living area and lot size
- Number of floors
- Waterfront and property view information
- Property condition
- Construction and renovation year
- City and location information
- House sale price (target variable)

**Target Variable:** `price`

## 🔍 Project Workflow

1. Data collection and loading
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature engineering
5. Machine Learning model training
6. Model comparison and evaluation
7. House price prediction
8. Interactive dashboard development

## 🚀 Future Enhancements

- Hyperparameter optimization
- Advanced regression algorithms
- Interactive geographical housing maps
- Real-time housing market data integration
- Model deployment through cloud platforms

## 👩‍💻 Author

Chamanpreet kaur

B.Tech(CSE) Student

Interested in Data Science, Artificial Intelligence, Machine Learning, and Data Analytics.

## 📄 License
This project is developed for educational and learning purposes. Please check the dataset's original license before redistribution.
