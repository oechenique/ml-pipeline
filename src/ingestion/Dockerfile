FROM python:3.9-slim

USER root

# Instalar dependencias del sistema, especialmente GDAL
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    g++ \
    libproj-dev \
    libsqlite3-dev \
    libjpeg-dev \
    libpng-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Configuración de variables de entorno para GDAL
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal
ENV GDAL_CONFIG=/usr/bin/gdal-config

WORKDIR /app

COPY src/ingestion/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

CMD ["python", "main.py"]
