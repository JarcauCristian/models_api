FROM python:3.11-alpine

WORKDIR /app

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        make \
        gcc \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install -r requirements.txt

ENV POSTGRES_USER = postgres
ENV POSTGRES_PASSWORD = postgres
ENV POSTGRES_HOST = 127.0.0.1
ENV POSTGRES_PORT = postgres
ENV POSTGRES_DB = postgres

ENV MLFLOW_TRACKING_USERNAME = username
ENV MLFLOW_TRACKING_PASSWORD = password
ENV AWS_ACCESS_KEY_ID = username
ENV AWS_SECRET_ACCESS_KEY = password
ENV MLFLOW_S3_ENDPOINT_URL = http://localhost

EXPOSE 8000

CMD ["python3", "main.py"]
