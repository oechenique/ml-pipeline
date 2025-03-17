# Usar la imagen oficial de Apache Spark con Python
FROM apache/spark-py:latest

# Cambiar a usuario root para instalar paquetes
USER root

# Instalar herramientas necesarias
RUN apt-get update && apt-get install -y wget dos2unix

# Crear directorio para Ivy
RUN mkdir -p /root/.ivy2 && chmod -R 777 /root/.ivy2

# Instalar librerías necesarias con permisos root
RUN pip install --no-cache-dir \
    'apache-sedona==1.4.1' \
    'pyspark==3.4.0' \
    'geopandas' \
    'shapely' \
    'h3' \
    'pandas' \
    'numpy' \
    'requests' \
    'matplotlib' \
    'jupyterlab' \
    'notebook'

# Configurar variables de entorno para Spark y Sedona
ENV SPARK_HOME="/opt/spark"
ENV SPARK_PYTHONPATH="${SPARK_HOME}/python"
ENV PYTHONPATH="${SPARK_PYTHONPATH}:${SPARK_HOME}/python/lib/py4j-0.10.9.7-src.zip"
ENV PYSPARK_PYTHON=python3
ENV SEDONA_VERSION="1.4.1"

# Descargar JARs de Sedona
RUN cd /opt/spark/jars && \
    wget -q https://repo1.maven.org/maven2/org/apache/sedona/sedona-python-adapter-3.4_2.12/${SEDONA_VERSION}/sedona-python-adapter-3.4_2.12-${SEDONA_VERSION}.jar && \
    wget -q https://repo1.maven.org/maven2/org/apache/sedona/sedona-viz-3.4_2.12/${SEDONA_VERSION}/sedona-viz-3.4_2.12-${SEDONA_VERSION}.jar && \
    wget -q https://repo1.maven.org/maven2/org/apache/sedona/sedona-sql-3.4_2.12/${SEDONA_VERSION}/sedona-sql-3.4_2.12-${SEDONA_VERSION}.jar && \
    wget -q https://repo1.maven.org/maven2/org/locationtech/jts/jts-core/1.18.0/jts-core-1.18.0.jar && \
    chmod 644 *.jar

# Configurar directorio de trabajo y permisos
RUN useradd -m -s /bin/bash -u 1001 jupyter && \
    mkdir -p /home/jupyter/.local && \
    mkdir -p /home/jupyter/.ivy2 && \
    mkdir -p /notebooks && \
    chown -R jupyter:0 /home/jupyter && \
    chown -R jupyter:0 /notebooks && \
    chmod -R 775 /home/jupyter && \
    chmod -R 775 /notebooks

# Copiar configuración de Spark
COPY spark-defaults.conf $SPARK_HOME/conf/spark-defaults.conf

# Copiar script de arranque
COPY entrypoint.sh /
RUN dos2unix /entrypoint.sh && chmod +x /entrypoint.sh

# Ajustar permisos
RUN chown -R jupyter:0 $SPARK_HOME && \
    chmod -R 755 $SPARK_HOME

# Configurar variables de entorno para el usuario
ENV HOME=/home/jupyter \
    SHELL=/bin/bash \
    USER=jupyter \
    NB_UID=1001

# Volver al usuario jupyter
USER jupyter
WORKDIR $HOME

# Exponer puertos
EXPOSE 4040 7077 8080 8888

# Iniciar contenedor
ENTRYPOINT ["/entrypoint.sh"]