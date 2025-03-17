#!/bin/bash

echo "🚀 Configurando Spark con Sedona..."

# Variables de entorno
export SPARK_HOME="/opt/spark"
export PYTHONPATH=$SPARK_HOME/python:$PYTHONPATH
export PATH=$SPARK_HOME/bin:$PATH

# Mostrar configuración de Spark
echo "📌 Configuración de Spark:"
cat $SPARK_HOME/conf/spark-defaults.conf

# Iniciar Jupyter Notebook
echo "📘 Iniciando Jupyter Notebook en http://localhost:8888"
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --notebook-dir=/notebooks