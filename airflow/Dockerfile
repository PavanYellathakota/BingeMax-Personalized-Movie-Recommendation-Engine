# File: airflow/Dockerfile
FROM apache/airflow:2.9.1-python3.11

USER root

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY airflow/requirements.txt /requirements.txt

USER airflow

RUN pip install --no-cache-dir apache-airflow-providers-sqlite apache-airflow-providers-fab
RUN pip install --no-cache-dir -r /requirements.txt

USER root
COPY ./airflow/dags /opt/airflow/dags
COPY ./airflow/plugins /opt/airflow/plugins
