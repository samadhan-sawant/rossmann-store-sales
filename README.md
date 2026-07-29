# rossmann-store-sales
End-to-end Machine Learning project for Rossmann Store Sales Forecasting using Python, Scikit-learn, Random Forest, XGBoost, and Feature Engineering.

# 📈 Rossmann Store Sales Forecasting

An end-to-end Machine Learning project to predict daily sales for Rossmann retail stores using historical sales data. The project demonstrates the complete ML workflow, including data cleaning, exploratory data analysis (EDA), feature engineering, model development, evaluation, and model comparison.

---

## 🚀 Project Overview

The objective of this project is to build a machine learning model that accurately forecasts daily store sales. Accurate sales forecasting helps businesses optimize inventory, staffing, promotional planning, and overall operational efficiency.

---

## 📊 Dataset

**Source:** Rossmann Store Sales Dataset (Kaggle)

The dataset contains:

- Store Information
- Daily Sales
- Customers
- Promotions
- School & State Holidays
- Competition Information
- Store Type
- Assortment
- Date

---

# 🛠 Project Workflow

## 1️⃣ Data Cleaning

- Handled missing values
- Checked duplicate records
- Corrected data types
- Treated invalid values

---

## 2️⃣ Exploratory Data Analysis (EDA)

Performed detailed analysis to understand business trends and customer behavior.

### Key Insights

- 📌 Promotions significantly increased sales.
- 📌 Sunday sales were lowest because many stores remained closed.
- 📌 Store Type B had the highest average sales, but represented only a small percentage of stores.
- 📌 Strong weekly and seasonal sales patterns were observed.
- 📌 Customer count showed a strong positive relationship with sales.

---

## 3️⃣ Feature Engineering

Created multiple features to improve model performance.

### Calendar Features

- Year
- Month
- Quarter
- Day of Week

### Time-Series Features

- Lag Features
- Rolling Mean Features

> Implemented `shift(1)` before rolling calculations to prevent **data leakage**.

---

## 4️⃣ Model Development

The following regression models were trained and compared:

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

---

## 📈 Model Evaluation

Models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

| Model | MAE | RMSE | R² Score |
|-------|------:|------:|------:|
| Linear Regression | 747.80 | 1058.12 | 0.919 |
| Random Forest | **471.09** | **735.52** | **0.961** |
| XGBoost | 486.98 | 743.18 | 0.960 |

**Best Model:** Random Forest Regressor

---

## 📉 Visualizations

The project includes:

- Sales Distribution
- Promotion Analysis
- Store Type Analysis
- Correlation Heatmap
- Actual vs Predicted Plot
- Residual Plot
- Feature Importance

---

## 🧰 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Jupyter Notebook

---


```

---

## 💡 Key Learnings

Through this project, I gained practical experience in:

- End-to-end Machine Learning workflow
- Business-oriented Exploratory Data Analysis
- Time-series feature engineering
- Preventing data leakage
- Model comparison and evaluation
- Building robust forecasting models

---

## 🔮 Future Improvements

- Hyperparameter tuning using Grid Search / Random Search
- FastAPI model deployment
- Docker containerization
- Time-series cross-validation
- MLflow experiment tracking
- CI/CD pipeline integration

---

## 👨‍💻 Author

**Samadhan Sawant**

- GitHub: https://github.com/samadhan-sawant
- LinkedIn:https://www.linkedin.com/in/samadhan-sawant-319b47190/

---

⭐ If you found this project useful, feel free to star this repository!
