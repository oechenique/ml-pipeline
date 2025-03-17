# Estructura del Proyecto

```
geospatial-ml-engineering/
├── .github/                      # Configuración de GitHub
│   └── workflows/                # GitHub Actions workflows
│       └── ci-cd.yml            # Pipeline de CI/CD
│
├── configs/                      # Configuraciones
│   ├── .env.example             # Template de variables de entorno
│   ├── config.yaml              # Configuración general
│   └── logging.yaml             # Configuración de logging
│
├── data/                        # Datos
│   ├── raw/                     # Datos crudos
│   ├── processed/               # Datos procesados
│   └── final/                   # Datos finales para visualización
│
├── docker/                      # Configuración de Docker
│   ├── api/                     # Dockerfile y configs para API
│   ├── spark/                   # Dockerfile y configs para Spark
│   └── streamlit/              # Dockerfile y configs para Streamlit
│
├── notebooks/                   # Jupyter notebooks
│   ├── exploration/            # Notebooks de exploración
│   ├── modeling/               # Notebooks de modelado
│   └── visualization/          # Notebooks de visualización
│
├── src/                        # Código fuente principal
│   ├── api/                    # API FastAPI
│   │   ├── database.py         # Configuración de base de datos
│   │   ├── endpoints.py        # Endpoints de la API
│   │   ├── models.py           # Modelos Pydantic
│   │   └── main.py            # Punto de entrada de la API
│   │
│   ├── ingestion/             # Pipeline de ingesta de datos
│   │   ├── ign_wfs.py         # Ingesta de datos IGN
│   │   ├── hdx_population.py  # Ingesta de datos de población
│   │   └── osm_data.py        # Ingesta de datos OSM
│   │
│   ├── models/                # Modelos de ML
│   │   ├── clustering.py      # Modelo de clustering
│   │   ├── feature_engineering.py  # Ingeniería de features
│   │   └── predictions.py     # Modelo predictivo
│   │
│   └── processing/            # Procesamiento de datos
│       ├── merge_datasets.py  # Integración de datasets
│       └── process_h3.py      # Procesamiento de grids H3
│
├── streamlit/                  # Dashboard Streamlit
│   ├── components/            # Componentes reutilizables
│   ├── pages/                 # Páginas del dashboard
│   └── main.py               # Punto de entrada del dashboard
│
├── tests/                     # Tests
│   ├── conftest.py           # Configuración de pytest
│   ├── test_api/             # Tests de API
│   ├── test_ingestion/       # Tests de ingesta
│   └── test_models/          # Tests de modelos
│
├── .env.example              # Template de variables de entorno
├── .gitignore               # Configuración de git ignore
├── docker-compose.yml       # Configuración de Docker Compose
├── README.md               # Documentación principal
└── requirements.txt        # Dependencias de Python
```

## Notas sobre la estructura

1. **Separación de Componentes**
   - Cada componente principal (API, ingesta, modelos) tiene su propio directorio
   - Los tests siguen la misma estructura que el código
   - Configuraciones separadas por ambiente

2. **Datos**
   - Estructura clara para el flujo de datos (raw → processed → final)
   - Separación de datos de entrenamiento y producción

3. **Documentación**
   - README principal para el proyecto
   - Documentación específica en cada componente
   - Notebooks para análisis y ejemplos

4. **Deployment**
   - Configuración completa de Docker
   - Pipeline de CI/CD con GitHub Actions
   - Variables de entorno manejadas apropiadamente

5. **Testing**
   - Tests organizados por componente
   - Configuración centralizada de pytest
   - Datos de prueba separados

6. **Dashboard**
   - Componentes modulares
   - Páginas separadas para mejor organización
   - Configuración independiente