import joblib
import sklearn
import pandas as pd
import joblib
from pathlib import Path
from fastapi import FastAPI , HTTPException,Request
from pydantic import BaseModel , ConfigDict
import sys
import datetime as dt
from contextlib import asynccontextmanager



project_dir = Path.cwd().parent

model_path = project_dir / "model" / "rossmann_model.pkl"
# db_path = project_dir / "data" / "rossmann.db"

# print("Model file exists:", model_path.exists())
# print("Database file exists:", db_path.exists())

# model = joblib.load(model_path)



# print("Model loaded:", type(model).__name__)
# print("Expected feature count:", model.n_features_in_)

# if hasattr(model, "feature_names_in_"):
#     print("Expected feature names:")
#     print(list(model.feature_names_in_))


sys.path.insert(0, str(project_dir / "app"))

from  pipeline import predict_sales

print("Pipeline imported successfully")
# print("Model loaded:", type(pipeline.model).__name__)


@asynccontextmanager
async def lifespan(app:FastAPI):
    print("Loading ML model")

    app.state.model = joblib.load(model_path)
    print("model loaded ")

    yield

    print("Shutting down application")









app = FastAPI(title='Sales Predication API',version="1.0",lifespan=lifespan)

class SaleInput(BaseModel):
    model_config=ConfigDict(extra="forbid")

    Store: int
    Date: dt.date
    Open: int
    Promo: int
    StateHoliday:str
    SchoolHoliday: int    

@app.get("/")
def home():
    return{"This is my first End to End ML project:😀"}
@app.get("/health")
def health():
    return{"Status":"Ok"}


@app.post("/predict")
def predit(data:SaleInput,request:Request):
     model=app.state.model
     result = predict_sales(model,data.model_dump(mode="json"))
     return(result)
     