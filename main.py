import os
import io
import json
import numpy
import base64
import mlflow
import decimal
import uvicorn
import requests
import tempfile
import numpy as np
import pandas as pd
from skl2onnx import to_onnx
from pydantic import BaseModel
from fastapi.responses import FileResponse
from starlette.responses import JSONResponse
from sklearn.preprocessing import LabelEncoder
from fastapi import FastAPI, UploadFile, Header
from mlflow import MlflowClient, MlflowException
from fastapi.middleware.cors import CORSMiddleware
from skl2onnx.common.data_types import FloatTensorType
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import create_engine, Column, String, DateTime, Numeric, Integer
from redis_cache import is_data_stale, get_data_from_redis, set_data_in_redis, update_timestamp

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
    target_column = Column(String)


password = str(os.getenv("POSTGRES_PASSWORD")).strip().replace("\n", "")

engine = create_engine(f'postgresql+psycopg2://'
                       f'{os.getenv("POSTGRES_USER")}:{password}@{os.getenv("POSTGRES_HOST")}'
                       f':{os.getenv("POSTGRES_PORT")}/{os.getenv("POSTGRES_DB")}')
Session = sessionmaker(bind=engine)


@app.get("/models")
async def connection_test():
    return JSONResponse(content="Server Works!", status_code=200)


@app.get("/models/model/user", tags=["GET"])
async def models_user(user_id: str, changed: bool = False, authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content="Unauthorized!")
    
    token = authorization.split(" ")[1]
    response = requests.get(os.getenv("KEYCLOAK_URL"), headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return JSONResponse(status_code=401, content="Unauthorized!")

    cache_key = f"models_list_{user_id}"
    if not is_data_stale(cache_key, 86400) and not changed:
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
            "score": 0.0 if int(row.score_count) == 0 else round(float(row.score) / float(row.score_count), 2)
        })

    set_data_in_redis(cache_key, models_list, 86400)
    update_timestamp(cache_key)

    session.close()

    return JSONResponse(content=models_list, status_code=200)


@app.get("/models/model/all", tags=["GET"])
async def models_all(changed: bool = False, authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content="Unauthorized!")
    
    token = authorization.split(" ")[1]
    response = requests.get(os.getenv("KEYCLOAK_URL"), headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return JSONResponse(status_code=401, content="Unauthorized!")

    cache_key = f"models_list"
    if not is_data_stale(cache_key, 86400) and not changed:
        cached_data = json.loads(get_data_from_redis(cache_key))
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
            "score": 0.0 if int(row.score_count) == 0 else round(float(row.score) / float(row.score_count), 2)
        })

    set_data_in_redis(cache_key, json.dumps(models_list), 86400)
    update_timestamp(cache_key)

    session.close()
    return JSONResponse(content=models_list, status_code=200)


@app.get("/models/model", tags=["GET"])
async def model(model_id: str, changed: bool = False, authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content="Unauthorized!")
    
    token = authorization.split(" ")[1]
    response = requests.get(os.getenv("KEYCLOAK_URL"), headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return JSONResponse(status_code=401, content="Unauthorized!")

    cache_key = f"model_data_{model_id}"
    if not is_data_stale(cache_key, 86400) and not changed:
        cached_data = json.loads(get_data_from_redis(cache_key))
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
        "target_column": "" if results[0].target_column is None else results[0].target_column,
        "score": round(float(results[0].score) / float(results[0].score_count), 2) if results[0].score_count > 0 else 0.0
    }

    set_data_in_redis(cache_key, json.dumps(model_data), 86400)
    update_timestamp(cache_key)

    session.close()
    return JSONResponse(content=model_data, status_code=200)


@app.post("/models/prediction", tags=["POST"])
async def prediction(model_id: str, file: UploadFile, authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content="Unauthorized!")
    
    token = authorization.split(" ")[1]
    response = requests.get(os.getenv("KEYCLOAK_URL"), headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return JSONResponse(status_code=401, content="Unauthorized!")

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

    number_of_features = [v for v in model_description["column_dtypes"] if v != results.target_column] if results.target_column is not None else len(model_description["column_dtypes"])

    if len(number_of_features) != len(df.columns):
        return JSONResponse(content="Column Number does not match!", status_code=400)
    
    if results.target_column is not None:
        numeric_columns = sum(1 for key, value in model_description["column_ranges"].items() if value is not None and key != results.target_column)
        categorical_columns = sum(1 for key, value in model_description["column_categories"].items() if value is not None and key != results.target_column)
        unique_identifiers_columns = sum(1 for key, value in model_description["column_unique_values"].items()
                                        if value is not None and key != results.target_column)
    else:
        numeric_columns = sum(1 for value in model_description["column_ranges"].values() if value is not None)
        categorical_columns = sum(1 for value in model_description["column_categories"].values() if value is not None)
        unique_identifiers_columns = sum(1 for value in model_description["column_unique_values"].values()
                                        if value is not None)

    columns = list(df.columns)

    numeric_count = 0
    categorical_count = 0
    unique_identifier_count = 0

    for column in columns:
        if results.target_column is not None and column == results.target_column:
            continue
        
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

    if categorical_count > 0 or unique_identifier_count > 0:
        label_encoder = LabelEncoder()
        for column in df.columns:
            if isinstance(df[column][0], str):
              df[column] = label_encoder.fit_transform(df[column])

    client = MlflowClient()
    run_id = client.get_registered_model(model_id).latest_versions[0].run_id
       
    sk_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    predictions = []
    for i, row in df.iterrows():
        prediction = sk_model.predict([np.array(row)])
        predictions.append(prediction)

    data = {}

    for i, row in enumerate(predictions):
        data[i] = round(float(row[0]), 2)

    return JSONResponse(content=data, status_code=200)

    
@app.get("/models/model/download", tags=["GET"])
async def download_model(model_id: str, authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content="Unauthorized!")
    
    token = authorization.split(" ")[1]
    response = requests.get(os.getenv("KEYCLOAK_URL"), headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return JSONResponse(status_code=401, content="Unauthorized!")

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    
    with Session() as session:
        results = session.query(MyTable).filter(model_id == MyTable.model_id).first()

        if results is None:
            return JSONResponse(content="Model ID Not Found!", status_code=404)

    client = MlflowClient()
    run_id = client.get_registered_model(model_id).latest_versions[0].run_id

    csv_data = json.loads(results.description)

    if results.target_column is not None:
        number_of_features = len([v for v in csv_data["column_dtypes"].keys() if v != results.target_column])
    else:
        number_of_features = len([v for v in csv_data["column_dtypes"].keys()])

    sk_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    initial_types = [('float_input', FloatTensorType([None, number_of_features]))]
    onnx_model = to_onnx(sk_model, initial_types=initial_types)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.onnx')
    temp_file_name = temp_file.name
    with open(temp_file_name, "wb") as f:
        f.write(onnx_model.SerializeToString())
    temp_file.close()

    return FileResponse(path=temp_file_name, media_type='application/octet-stream', filename=f"{results.model_name}.onnx", 
                        headers={"Content-Disposition": f"attachment; filename={results.model_name}.onnx"})


@app.get("/models/model_details", tags=["GET"])
async def model_details(model_id: str, authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content="Unauthorized!")
    
    token = authorization.split(" ")[1]
    response = requests.get(os.getenv("KEYCLOAK_URL"), headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return JSONResponse(status_code=401, content="Unauthorized!")

    cache_key = f"model_details_{model_id}"
    if not is_data_stale(cache_key, 86400):
        cached_data = json.loads(get_data_from_redis(cache_key))
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

    set_data_in_redis(cache_key, json.dumps(model_details_data), 86400)
    update_timestamp(cache_key)

    return JSONResponse(content=model_details_data, status_code=200)


@app.get("/models/model_images", tags=["GET"])
async def model_images(model_id: str, authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content="Unauthorized!")
    
    token = authorization.split(" ")[1]
    response = requests.get(os.getenv("KEYCLOAK_URL"), headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return JSONResponse(status_code=401, content="Unauthorized!")

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
                images[str(artifact.path).split("/")[-1]] = (f"data:image/png;base64,{(base64.b64encode(buffered.getvalue()).decode('utf-8'))}")
    except MlflowException as e:
        print(e)
        return JSONResponse(status_code=500, content="Could not get all the images!")

    return JSONResponse(content=images, status_code=200)


@app.get("/models/model_score", tags=["GET"])
async def model_score(model_id: str, authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content="Unauthorized!")
    
    token = authorization.split(" ")[1]
    response = requests.get(os.getenv("KEYCLOAK_URL"), headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return JSONResponse(status_code=401, content="Unauthorized!")

    with Session() as session:
        result = session.query(MyTable).filter(model_id == MyTable.model_id).first()

        if result is None:
            return JSONResponse(content="Model ID Not Found!", status_code=404)

    return JSONResponse(content= 0.0 if int(result.score_count) == 0 else round(float(result.score) / float(result.score_count), 2), status_code=200)


@app.post("/models/update_score", tags=["POST"])
async def update_score(score: Score, authorization: str = Header(None)):
    if authorization is None or not authorization.startswith("Bearer "):
        return JSONResponse(status_code=401, content="Unauthorized!")
    
    token = authorization.split(" ")[1]
    response = requests.get(os.getenv("KEYCLOAK_URL"), headers={"Authorization": f"Bearer {token}"})
    if response.status_code != 200:
        return JSONResponse(status_code=401, content="Unauthorized!")

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
    os.environ["MLFLOW_TRACKING_INSECURE_TLS"] = "true"
    os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = "1000"
    uvicorn.run(app, host='0.0.0.0')
