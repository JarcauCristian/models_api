import json
import os
import mlflow
import numpy
import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile
from dotenv import load_dotenv
from mlflow import MlflowClient
from starlette.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, String, DateTime, Numeric, Integer

app = FastAPI()

load_dotenv()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

Base = declarative_base()


class MyTable(Base):
    __tablename__ = "models"
    model_id = Column(String, primary_key=True)
    user_id = Column(String)
    created_at = Column(DateTime)
    description = Column(String)
    score = Column(Numeric)
    model_name = Column(String)
    score_count = Column(Integer)


engine = create_engine(f'postgresql+psycopg2://'
                       f'{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@{os.getenv("POSTGRES_HOST")}'
                       f':{os.getenv("POSTGRES_PORT")}/{os.getenv("POSTGRES_DB")}')
Session = sessionmaker(bind=engine)


@app.get("/")
async def connection_test():
    return JSONResponse(content="Server Works!", status_code=200)


@app.get("/models")
async def models():
    session = Session()

    results = session.query(MyTable).all()

    models_list = []

    if len(results) == 0:
        session.close()
        return JSONResponse(content="No models found", status_code=404)

    for row in results:
        models_list.append({
            "model_id": row.model_id,
            "model_name": row.model_name,
            "description": row.description,
            "created_at": str(row.created_at),
            "score": round(float(row.score / row.score_count), 2)
        })

    session.close()
    return JSONResponse(content=models_list, status_code=200)


@app.get("/model")
async def model(model_id: str):
    session = Session()

    results = session.query(MyTable).filter(MyTable.model_id == model_id).all()

    if len(results) == 0:
        session.close()
        return JSONResponse(content="No model found with that ID.", status_code=404)

    session.close()

    model_data = {
        "model_id": results[0].model_id,
        "model_name": results[0].model_name,
        "description": results[0].description,
        "created_at": str(results[0].created_at),
        "score": round(float(float(results[0].score) / float(results[0].score_count)), 2) if float(results[0].score_count) > 0 else 0
    }

    return JSONResponse(content=model_data, status_code=200)


@app.post("/prediction")
async def prediction(model_id: str, file: UploadFile):
    if file.content_type != "text/csv":
        return JSONResponse(content="Only CSV files Allowed!", status_code=400)

    os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = "1000"

    mlflow.set_tracking_uri("https://mlflow.sedimark.work")

    session = Session()

    results = session.query(MyTable).filter(MyTable.model_id == model_id).first()

    if results is None:
        session.close()
        return JSONResponse(content="Model ID Not Found!", status_code=404)

    model_description = json.loads(results.description)

    session.close()

    df = pd.read_csv(file.file)

    if len(model_description["column_dtypes"]) != len(df.columns):
        return JSONResponse(content="Column Number does not match", status_code=400)

    numeric_columns = sum(1 for value in model_description["column_ranges"].values() if value is not None)
    categorical_columns = sum(1 for value in model_description["column_categories"].values() if value is not None)
    unique_identifiers_columns = sum(1 for value in model_description["column_unique_values"].values() if value is not None)

    columns = list(df.columns)

    numeric_count = 0
    categorical_count = 0
    unique_identifier_count = 0

    for column in columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_count += 1
        else:
            if len(df[column].unique()) == len(df):
                unique_identifier_count += 1
            else:
                categorical_count += 1

    if numeric_columns != numeric_count or categorical_columns != categorical_count or unique_identifiers_columns != unique_identifier_count:
        return JSONResponse(content="Dataset format does not match the model input!", status_code=400)

    sk_model = mlflow.sklearn.load_model(f"runs:/{model_id}/model")

    predictions: numpy.ndarray = sk_model.predict(df)
    data = {}

    for i, row in enumerate(predictions):
        data[i] = round(float(row), 2)

    return JSONResponse(content=data, status_code=200)


@app.get("/model_details")
async def model_details(model_id: str):
    session = Session()

    results = session.query(MyTable).filter(MyTable.model_id == model_id).first()

    if results is None:
        session.close()
        return JSONResponse(content="Model ID Not Found!", status_code=404)

    session.close()

    client = MlflowClient(tracking_uri="https://mlflow.sedimark.work")
    data = client.get_run(model_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}

    return JSONResponse(content={"params": data.params, "metrics": data.metrics, "tags": tags}, status_code=200)


@app.get("/model_score")
async def model_score(model_id: str):
    session = Session()

    result = session.query(MyTable).filter(MyTable.model_id == model_id).first()

    if result is None:
        session.close()
        return JSONResponse(content="Model ID Not Found!", status_code=404)

    return JSONResponse(content=round(result.score / result.score_count, 2), status_code=200)


@app.post("/update_score")
async def update_score(model_id: str, score: float):
    if score < 0 or score > 10:
        return JSONResponse(content="Score should be between 0 and 10!", status_code=400)

    session = Session()

    result = session.query(MyTable).filter(MyTable.model_id == model_id).first()

    if result is None:
        session.close()
        return JSONResponse(content="Model ID Not Found!", status_code=404)

    result.score += score

    result.score_count += 1

    session.commit()
    session.close()

    return JSONResponse(content="Score updated successfully!", status_code=200)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0')
