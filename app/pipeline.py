from pathlib import Path
import sqlite3
# import joblib
import pandas as pd

# Project folder
PROJECT_DIR = Path(__file__).resolve().parents[1]

# Paths
MODEL_PATH = PROJECT_DIR / "model" / "rossmann_model.pkl"
DB_PATH = PROJECT_DIR / "data" / "rossmann.db"

# Load model once
# model = joblib.load(MODEL_PATH)

def get_db_connection():
    return sqlite3.connect(DB_PATH)




def jason_validate(jason_data):
    missing_col = []

    for col in REQUIRED_COLUMNS:
        if col not in jason_data:
            missing_col.append(col)

    if missing_col:
        raise ValueError(f"Required column is missing: {missing_col}")

    if not isinstance(jason_data["Store"], int) or isinstance(jason_data["Store"], bool):
        raise ValueError("Store must be an integer")

    if jason_data["Promo"] not in [0, 1]:
        raise ValueError("Promo must be either 0 or 1")

    if jason_data["SchoolHoliday"] not in [0, 1]:
        raise ValueError("SchoolHoliday must be either 0 or 1")

    if jason_data["Open"] not in [0, 1]:
        raise ValueError("Open must be either 0 or 1")

    valid_holidays = ["0", "a", "b", "c"]

    if jason_data["StateHoliday"] not in valid_holidays:
        raise ValueError("Invalid StateHoliday value")

    try:
        pd.to_datetime(jason_data["Date"], errors="raise")
    except (ValueError, TypeError):
        raise ValueError("Date must be a valid date")

    clean_data = {
        column: jason_data[column]
        for column in REQUIRED_COLUMNS
    }

    return clean_data


REQUIRED_COLUMNS = [
    "Store",
    "Date",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday"
]




def get_store_info(data):
    validate_df = pd.DataFrame([jason_validate(data)])
    store_id = validate_df["Store"].iloc[0].item()

    with get_db_connection() as conn:
        store_df = pd.read_sql_query(
            "SELECT * FROM store_info WHERE Store = ?",
            conn,
            params=(store_id,)
        )

    if store_df.empty:
        raise ValueError("Store not found")

    merged_df = validate_df.merge(
        store_df,
        on="Store",
        how="left"
    )

    return merged_df



def create_date_features(merged_df):
    merged_df = merged_df.copy()
    merged_df["Date"] = pd.to_datetime(merged_df["Date"])

    merged_df["Month"] = merged_df["Date"].dt.month
    merged_df["Quarter"] = merged_df["Date"].dt.quarter
    merged_df["WeekOfYear"] = (
        merged_df["Date"].dt.isocalendar().week.astype(int)
    )
    merged_df["DayOfWeek"] = merged_df["Date"].dt.dayofweek + 1
    merged_df["IsWeekend"] = (
        merged_df["DayOfWeek"] >= 6
    ).astype(int)
    merged_df["Year"] = merged_df["Date"].dt.year

    return merged_df


def get_sales_history(df):
    df = df.copy()

    store_id = df["Store"].iloc[0].item()
    prediction_date = pd.to_datetime(
        df["Date"].iloc[0]
    ).strftime("%Y-%m-%d")

    with get_db_connection() as conn:
        history = pd.read_sql_query(
            """
            SELECT Store, Date, Sales
            FROM sales_history
            WHERE Store = ?
              AND Date < ?
            ORDER BY Date
            """,
            conn,
            params=(store_id, prediction_date)
        )

    if len(history) < 28:
        raise ValueError(
            f"Insufficient historical data for Store {store_id}. "
            f"Required: 28 records, Available: {len(history)}"
        )

    return history



def create_historical_features(df, history):
    df = df.copy()
    history = history.copy()

    history["Date"] = pd.to_datetime(history["Date"])

    history = history.sort_values(
        ["Store", "Date"]
    ).reset_index(drop=True)

    # Lag features
    df["lag_1"] = history["Sales"].iloc[-1]
    df["lag_7"] = history["Sales"].iloc[-7]
    df["lag_14"] = history["Sales"].iloc[-14]
    df["lag_28"] = history["Sales"].iloc[-28]

    # Rolling sales averages
    df["Rolling_7"] = history["Sales"].tail(7).mean()
    df["Rolling_14"] = history["Sales"].tail(14).mean()
    df["Rolling_28"] = history["Sales"].tail(28).mean()

    return df



def encode_features(df):
    df = df.copy()

    model_columns = [
        "Store", "DayOfWeek", "Open", "Promo",
        "SchoolHoliday", "CompetitionDistance", "Promo2", "Year",
        "Month", "Quarter", "lag_1", "lag_7", "lag_14", "lag_28",
        "Rolling_7", "Rolling_14", "Rolling_28",
        "StateHoliday_a", "StateHoliday_b", "StateHoliday_c",
        "StoreType_a", "StoreType_b", "StoreType_c", "StoreType_d",
        "Assortment_a", "Assortment_b", "Assortment_c"
    ]

    df = pd.get_dummies(
        df,
        columns=["StoreType", "StateHoliday", "Assortment"],
        dtype=int
    )

    df = df.reindex(
        columns=model_columns,
        fill_value=0
    )

    return df



def predict_sales(model,data):
    df = get_store_info(data)
    df = create_date_features(df)
    history = get_sales_history(df)
    df = create_historical_features(df, history)
    df = encode_features(df)

    prediction = model.predict(df)

    return float(prediction[0])
