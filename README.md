# Rossmann Store Sales Forecasting

An end-to-end Machine Learning project to predict daily sales for Rossmann retail stores using Python, Scikit-learn, Random Forest, XGBoost, feature engineering, and FastAPI.

## Project Overview

The objective is to build a machine learning model that forecasts daily store sales using historical sales data.

Accurate sales forecasting helps businesses optimize inventory, staffing, promotional planning, and operational efficiency.

## Dataset

**Source:** Rossmann Store Sales Dataset (Kaggle)

The dataset contains:
- Store Information
- Daily Sales and Customers
- Promotions
- School and State Holidays
- Competition Information
- Store Type and Assortment
- Date

## Project Workflow

### 1. Data Cleaning
- Handled missing values
- Checked duplicate records
- Corrected data types
- Treated invalid values

### 2. Exploratory Data Analysis (EDA)

Key insights:
- Promotions significantly increased sales.
- Sunday sales were lowest because many stores remained closed.
- Store Type B had the highest average sales.
- Strong weekly and seasonal sales patterns were observed.
- Customer count showed a strong positive relationship with sales.

### 3. Feature Engineering

**Calendar Features**
- Year
- Month
- Quarter
- Day of Week

**Time-Series Features**
- Lag Features
- Rolling Mean Features

Implemented `shift(1)` before rolling calculations to prevent data leakage.

### 4. Model Development

Regression models trained and compared:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

### 5. Model Evaluation

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 747.80 | 1058.12 | 0.919 |
| Random Forest | **471.09** | **735.52** | **0.961** |
| XGBoost | 486.98 | 743.18 | 0.960 |

**Best Model:** Random Forest Regressor

## FastAPI Deployment

The trained machine learning model is integrated with FastAPI to expose sales predictions through a REST API.

### Deployment Architecture

1. Client sends store and prediction details through an API request.
2. FastAPI validates the input using Pydantic.
3. The trained model is loaded during application startup using the FastAPI Lifespan mechanism.
4. The prediction pipeline retrieves store information and historical sales from SQLite.
5. Feature engineering and encoding are performed.
6. The model generates the predicted sales value.
7. The API returns the prediction.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Home endpoint |
| GET | `/health` | Health check |
| POST | `/predict` | Predict store sales |

### Example Prediction Request

`POST /predict`

```json
{
  "Store": 1,
  "Date": "2015-07-31",
  "Open": 1,
  "Promo": 1,
  "StateHoliday": "0",
  "SchoolHoliday": 1
}
```

The API returns the predicted sales value.

### Running the API Locally

**1. Install dependencies**

```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib
```

**2. Place the required files**

The following files must be available locally:

```text
model/rossmann_model.pkl
data/rossmann.db
```

These large files are excluded from GitHub through `.gitignore`.

**3. Start the API**

From the project root:

```bash
cd Notebook
uvicorn inspect_model:app --reload
```

**4. Open Swagger documentation**

```text
http://127.0.0.1:8000/docs
```

Use the interactive Swagger interface to test the prediction endpoint.

## Project Structure

```text
rossmann-store-sales/
│
├── app/
│   └── pipeline.py
│
├── Notebook/
│   ├── inspect_model.py
│   └── rossmann-store-sales.ipynb
│
├── model/                  # Local model file
├── data/                   # Local SQLite database
│
├── .gitignore
└── README.md
```

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- SQLite
- FastAPI
- Pydantic
- Uvicorn
- Jupyter Notebook
- Git and GitHub

## Key Learnings

- End-to-end Machine Learning workflow
- Business-oriented EDA
- Time-series feature engineering
- Preventing data leakage
- Model comparison and evaluation
- Building prediction pipelines
- REST API development using FastAPI
- Model loading using application Lifespan
- Input validation and API testing

## Future Improvements

- Hyperparameter tuning
- Docker containerization
- Time-series cross-validation
- MLflow experiment tracking
- CI/CD pipeline integration
- Model optimization and performance monitoring

## Author

**Samadhan Sawant**

- GitHub: https://github.com/samadhan-sawant
- LinkedIn: https://www.linkedin.com/in/samadhan-sawant-319b47190/

⭐ If you found this project useful, feel free to star this repository!