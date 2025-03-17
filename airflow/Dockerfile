FROM apache/airflow:2.7.1

USER root
# Instalar dependencias de sistema
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq-dev \
        gcc \
        openjdk-11-jdk \
        libgdal-dev \
        python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Instalar dependencias de Python
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir \
        apache-airflow-providers-postgres \
        apache-airflow-providers-apache-spark \
        h3 \
        h3pandas \
        prophet \
        pulp \
        xgboost \
        mlflow \
        geopandas \
        shapely \
        streamlit \
        keplergl

# Configuración de Java para PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64