import os
import io
import json
import numpy
import joblib
import base64
import mlflow
import decimal
import uvicorn
import requests
import tempfile
import pandas as pd
from pydantic import BaseModel
from starlette.responses import JSONResponse
from mlflow import MlflowClient, MlflowException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, Header, Response, Request
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, String, DateTime, Numeric, Integer
from redis_cache import is_data_stale, get_data_from_redis, set_data_in_redis, update_timestamp
from dotenv import load_dotenv

load_dotenv(".env")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

Base = declarative_base()


class Score(BaseModel):
    model_id: str
    score: float


class MyTable(Base):
    __tablename__ = "models"
    model_id = Column(String, primary_key=True)
    user_id = Column(String)
    created_at = Column(DateTime)
    description = Column(String)
    score = Column(Numeric)
    model_name = Column(String)
    score_count = Column(Integer)
    dataset_user = Column(String)
    notebook_type = Column(String)


password = str(os.getenv("POSTGRES_PASSWORD")).strip().replace("\n", "")

engine = create_engine(f'postgresql+psycopg2://'
                       f'{os.getenv("POSTGRES_USER")}:{password}@{os.getenv("POSTGRES_HOST")}'
                       f':{os.getenv("POSTGRES_PORT")}/{os.getenv("POSTGRES_DB")}')
Session = sessionmaker(bind=engine)


@app.middleware("http")
async def middleware(request: Request, call_next):
    if request.headers.get("Authorization") is None or not request.headers.get("Authorization").startswith("Bearer "):
        return JSONResponse(status_code=401, content="You are unauthorized!")
    
    token = request.headers.get("Authorization").split(" ")[1]
    token_response = requests.get(os.getenv("KEYCLOAK_URL"), headers={"Authorization": f"Bearer {token}"})
    if token_response.status_code != 200:
        return JSONResponse(status_code=401, content="You are unauthorized!")
    
    response = await call_next(request)
    return response


@app.get("/models")
async def connection_test():
    return JSONResponse(content="Server Works!", status_code=200)


@app.get("/models/model/user", tags=["GET"])
async def models(user_id: str):
    cache_key = f"models_list_{user_id}"
    if not is_data_stale(cache_key, 600):
        cached_data = get_data_from_redis(cache_key)
        if cached_data:
            return JSONResponse(content=cached_data, status_code=200)

    session = Session()

    results = session.query(MyTable).filter(user_id == MyTable.user_id).all()

    if not results:
        return JSONResponse(content="No models found", status_code=404)

    models_list = []
    for row in results:
        models_list.append({
            "model_id": row.model_id,
            "model_name": row.model_name,
            "description": row.description,
            "created_at": str(row.created_at),
            "notebook_type": row.notebook_type,
            "score": round(float(row.score / row.score_count), 2)
        })

    set_data_in_redis(cache_key, models_list, 600)
    update_timestamp(cache_key)

    session.close()

    return JSONResponse(content=models_list, status_code=200)


@app.get("/models/model/all", tags=["GET"])
async def models():
    cache_key = f"models_list"
    if not is_data_stale(cache_key, 600):
        cached_data = get_data_from_redis(cache_key)
        if cached_data:
            return JSONResponse(content=cached_data, status_code=200)
        
    session = Session()

    results = session.query(MyTable).all()

    if not results:
        return JSONResponse(content="No models found", status_code=404)
    
    models_list = []
    for row in results:
        models_list.append({
            "model_id": row.model_id,
            "user_id": row.user_id,
            "dataset_user": row.dataset_user,
            "model_name": row.model_name,
            "description": row.description,
            "created_at": str(row.created_at),
            "notebook_type": row.notebook_type,
            "score": round(float(row.score / row.score_count), 2)
        })

    set_data_in_redis(cache_key, models_list, 600)
    update_timestamp(cache_key)

    session.close()
    return JSONResponse(content=models_list, status_code=200)


@app.get("/models/model", tags=["GET"])
async def model(model_id: str):
    cache_key = f"model_details_{model_id}"
    if not is_data_stale(cache_key, 3600):
        cached_data = get_data_from_redis(cache_key)
        if cached_data:
            return JSONResponse(content=cached_data, status_code=200)

    session = Session()

    results = session.query(MyTable).filter(model_id == MyTable.model_id).all()

    if not results:
        return JSONResponse(content="No model found with that ID.", status_code=404)
    
    model_data = {
        "model_id": results[0].model_id,
        "model_name": results[0].model_name,
        "description": results[0].description,
        "created_at": str(results[0].created_at),
        "notebook_type": results[0].notebook_type,
        "score": round(float(results[0].score) / float(results[0].score_count), 2) if results[0].score_count > 0 else 0
    }

    set_data_in_redis(cache_key, model_data, 3600)
    update_timestamp(cache_key)

    session.close()
    return JSONResponse(content=model_data, status_code=200)


@app.post("/models/prediction", tags=["POST"])
async def prediction(model_id: str, file: UploadFile):
    if file.content_type != "text/csv":
        return JSONResponse(content="Only CSV files Allowed!", status_code=400)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    session = Session()

    results = session.query(MyTable).filter(model_id == MyTable.model_id).first()

    if results is None:
        session.close()
        return JSONResponse(content="Model ID Not Found!", status_code=404)

    model_description = json.loads(str(results.description))

    session.close()

    df = pd.read_csv(file.file)

    if len(model_description["column_dtypes"]) != len(df.columns):
        return JSONResponse(content="Column Number does not match", status_code=400)

    numeric_columns = sum(1 for value in model_description["column_ranges"].values() if value is not None)
    categorical_columns = sum(1 for value in model_description["column_categories"].values() if value is not None)
    unique_identifiers_columns = sum(1 for value in model_description["column_unique_values"].values()
                                     if value is not None)

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

    if (numeric_columns != numeric_count or categorical_columns != categorical_count or unique_identifiers_columns !=
            unique_identifier_count):
        return JSONResponse(content="Dataset format does not match the model input!", status_code=400)

    client = MlflowClient()
    run_id = client.get_registered_model(model_id).latest_versions[0].run_id
       
    sk_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    predictions: numpy.ndarray = sk_model.predict(df)
    data = {}

    for i, row in enumerate(predictions):
        data[i] = round(float(row), 2)

    return JSONResponse(content=data, status_code=200)

    
@app.get("/models/download", tags=["GET"])
async def download_model(model_id: str):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    with Session() as session:
        results = session.query(MyTable).filter(model_id == MyTable.model_id).first()

        if results is None:
            return JSONResponse(content="Model ID Not Found!", status_code=404)

    client = MlflowClient()
    run_id = client.get_registered_model(model_id).latest_versions[0].run_id

    sk_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    with tempfile.TemporaryDirectory() as temp_dir:
        joblib.dump(sk_model, os.path.join(temp_dir, f"{results.model_name}.pkl"))
        model_filename = os.path.join(temp_dir, f"{results.model_name}.pkl")

        return Response(content=open(model_filename, "rb").read(), media_type="application/octet-stream")


@app.get("/models/model_details", tags=["GET"])
async def model_details(model_id: str):
    cache_key = f"model_details_{model_id}"
    if not is_data_stale(cache_key, 3600):
        cached_data = get_data_from_redis(cache_key)
        if cached_data:
            return JSONResponse(content=cached_data, status_code=200)

    with Session() as session:
        result = session.query(MyTable).filter(model_id == MyTable.model_id).first()

        if result is None:
            return JSONResponse(content="Model ID Not Found!", status_code=404)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    client = MlflowClient()
    registered_model = client.get_registered_model(model_id)
    run_id = registered_model.latest_versions[0].run_id
    description = registered_model.description
    data = client.get_run(run_id).data
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}

    model_details_data = {"params": data.params, "metrics": data.metrics, "tags": tags, "description": description}

    set_data_in_redis(cache_key, model_details_data, 3600)
    update_timestamp(cache_key)

    return JSONResponse(content=model_details_data, status_code=200)


@app.get("/models/model_images", tags=["GET"])
async def model_images(model_id: str):
    with Session() as session:
        result = session.query(MyTable).filter(model_id == MyTable.model_id).first()

        if result is None:
            return JSONResponse(content="Model ID Not Found!", status_code=404)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    client = MlflowClient()
    run_id = client.get_registered_model(model_id).latest_versions[0].run_id
    images = {}
    try:
        artifacts = client.list_artifacts(run_id)
        for artifact in artifacts:
            if ".png" in artifact.path:
                image = mlflow.artifacts.load_image(f"runs:/{run_id}/{artifact.path}")
                buffered = io.BytesIO()
                image_format = image.format if image.format else 'PNG'
                image.save(buffered, format=image_format)
                images[str(artifact.path).split("/")[-1]] = (f"data:image/png;base64,"
                                                             f"{(base64.b64encode(buffered.getvalue()).decode('utf-8'))}")
    except MlflowException:
        return JSONResponse(status_code=500, content="Could not get all the images!")

    return JSONResponse(content=images, status_code=200)


@app.get("/models/model_score", tags=["GET"])
async def model_score(model_id: str):
    with Session() as session:
        result = session.query(MyTable).filter(model_id == MyTable.model_id).first()

        if result is None:
            return JSONResponse(content="Model ID Not Found!", status_code=404)

    return JSONResponse(content=round(float(result.score / result.score_count), 2), status_code=200)


@app.post("/models/update_score", tags=["POST"])
async def update_score(score: Score):
    if score.score < 0 or score.score > 10:
        return JSONResponse(content="Score should be between 0 and 10!", status_code=400)

    with Session() as session:
        result = session.query(MyTable).filter(score.model_id == MyTable.model_id).first()

        if result is None:
            session.close()
            return JSONResponse(content="Model ID Not Found!", status_code=404)

        result.score += decimal.Decimal(score.score)

        result.score_count += 1

        session.commit()

    return JSONResponse(content="Score updated successfully!", status_code=200)


if __name__ == '__main__':
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ['MLFLOW_TRACKING_PASSWORD'] = str(os.getenv("MLFLOW_TRACKING_PASSWORD")).strip().replace("\n", "")
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ['AWS_SECRET_ACCESS_KEY'] = str(os.getenv("AWS_SECRET_ACCESS_KEY")).strip().replace("\n", "")
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
    os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = "1000"
    uvicorn.run(app, host='0.0.0.0')
