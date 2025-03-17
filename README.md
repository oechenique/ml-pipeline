# 🚲 Bike-Sharing Geospatial ML Pipeline | 自転車シェアリング地理空間MLパイプライン


## 🌟 Overview | 概要

A comprehensive ML pipeline for bike-sharing network optimization using geospatial analysis with real-time data processing. This project integrates data engineering, spatial analytics, machine learning, and interactive visualization to provide actionable insights for bike-sharing operations.

### Key Features | 主な機能

- **Geospatial Analysis**: H3 hexagonal grid indexing for uniform spatial analysis
- **Clustering**: Automated station clustering based on usage patterns
- **Demand Prediction**: ML models to forecast hourly bike demand
- **Interactive Dashboard**: Real-time visualization of stations, clusters, and predictions
- **Full Data Pipeline**: End-to-end workflow from data extraction to visualization

## 🔍 Project Components | プロジェクト構成

### 1. Data Pipeline Architecture | データパイプラインアーキテクチャ

The project is built on a modular architecture with containerized components:

- **Data Ingestion**: Extracts bike trip data from Google BigQuery
- **Spatial Processing**: Indexes stations using Uber's H3 grid system
- **Storage Layer**: PostgreSQL with PostGIS for geospatial data storage
- **ML Pipeline**: Clustering, prediction, and time series analysis
- **Visualization Layer**: Interactive Streamlit dashboard

#### Architecture

```
                       ┌─────────────────┐
                       │                 │
           ┌───────────┤  Data Ingestion ├───────────┐
           │           │                 │           │
           │           └─────────────────┘           │
           │                                         │
           │                                         │
           ▼                                         │
┌─────────────────────┐                   ┌─────────────────────┐
│                     │                   │                     │
│ Processing & Storage│                   │ Automation &        │
│                     │                   │ Deployment          │
└──────────┬──────────┘                   └─────────────────────┘
           │                                         ▲
           │                                         │
           │                                         │
           ▼                                         │
┌─────────────────────┐                   ┌─────────────────────┐
│                     │                   │                     │
│   Machine Learning  ├───────────────────▶   Visualization     │
│                     │                   │                     │
└─────────────────────┘                   └─────────────────────┘
```

### 2. Technologies Used | 使用技術

| Component | Technologies |
|-----------|--------------|
| **Data Engineering** | Apache Airflow, Docker, PostgreSQL/PostGIS |
| **Geospatial Analysis** | H3 Grid System, GeoPandas, Shapely |
| **Machine Learning** | Scikit-learn, Clustering, Time Series Analysis |
| **Visualization** | Streamlit, Folium, Plotly, Kepler.gl |
| **Deployment** | Docker Compose, CI/CD Pipeline |

### 3. ML Models Implemented | 実装されたMLモデル

- **Spatial Clustering**: Identifies station groups with similar characteristics
- **Demand Prediction**: Forecasts hourly bike demand per station
- **Time Series Analysis**: Analyzes temporal patterns and trends

## 🚀 Getting Started | はじめに

### Prerequisites | 前提条件

- Docker and Docker Compose
- Git
- 8GB RAM (minimum)
- 20GB free disk space

### Installation & Deployment | インストールとデプロイ

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bike-sharing-ml.git
   cd bike-sharing-ml
   ```

2. Start the containers:
   ```bash
   docker-compose up -d db_service
   # Wait for DB to initialize
   docker-compose up -d
   ```

3. Initialize Airflow:
   ```bash
   docker exec -it airflow-webserver airflow db init
   docker exec -it airflow-webserver airflow users create \
     --username admin \
     --password admin \
     --firstname Admin \
     --lastname User \
     --role Admin \
     --email admin@example.com
   ```

4. Access the services:
   - Airflow UI: http://localhost:8080
   - Streamlit Dashboard: http://localhost:8501
   - PostgreSQL: localhost:5432

### DAG Execution Order | DAG実行順序

1. `bike_sharing_ingestion`: Extracts data from BigQuery or generates synthetic data
2. `bike_sharing_postgres_storage`: Stores the data in PostgreSQL
3. `bike_sharing_h3_indexing`: Generates H3 indices for stations
4. `bike_sharing_clustering`: Performs station clustering
5. `bike_sharing_prediction`: Builds and runs demand prediction model
6. `bike_sharing_time_series`: Analyzes temporal patterns
7. `launch_streamlit_dashboard`: Launches the interactive dashboard

## 📊 Dashboard Features | ダッシュボード機能

The Streamlit dashboard provides:

- Interactive map with H3 hexagons and stations colored by cluster
- Filtering by city and cluster
- Demand prediction visualization by hour and day
- Top/bottom stations by predicted demand
- Cluster distribution analytics
- Model performance metrics

## 🔧 Technical Implementation | 技術的実装

### Geospatial Indexing with H3 | H3による地理空間インデックス

We implemented Uber's H3 hierarchical hexagonal grid system which provides:
- Uniform spatial analysis (hexagons have equal area)
- Multi-resolution capabilities (from ~1km² to ~1m²)
- Efficient spatial joins and hierarchical clustering

```python
# Sample H3 indexing code
def index_stations_with_h3(df, resolution=9):
    df['h3_index'] = df.apply(
        lambda row: h3.geo_to_h3(row.latitude, row.longitude, resolution), 
        axis=1
    )
    return df
```

### Machine Learning Pipeline | 機械学習パイプライン

The ML pipeline includes:

1. **Feature Engineering**:
   - Temporal features (hour of day, day of week)
   - Weather impact features
   - Spatial features (H3 neighborhood analysis)

2. **Clustering Algorithm**:
   - K-means with optimal cluster determination
   - Spatial validation using H3 grid

3. **Demand Prediction**:
   - Random Forest Regression
   - Time-based cross-validation
   - Feature importance analysis

### PostgreSQL Schema Design | PostgreSQLスキーマ設計

The database schema is optimized for geospatial queries:

- Partitioned tables for efficient querying of historical data
- PostGIS geometry columns for spatial operations
- Indices on frequently queried columns
- H3 indices stored for rapid spatial joins

## 📈 Results and Performance | 結果とパフォーマンス

- **Processed Data**: 2.5M+ historical bike trips spanning 2013-2023
- **Stations Analyzed**: 1,296 unique stations across multiple cities
- **Clustering Performance**: Identified 3 distinct station clusters with > 85% silhouette score
- **Prediction Accuracy**: Achieved 0.60 R² and 2.29 RMSE on demand predictions
- **Processing Time**: Full pipeline execution in < 10 minutes

## 🔮 Future Work | 今後の展望

- Integration with real-time weather data
- Expansion to multi-city comparative analysis
- Rebalancing optimization algorithms
- Mobile app integration
- A/B testing framework for demand model validation

## 💻 Tech Stack | 技術スタック
- **Backend**: Python, Airflow, PostgreSQL, PostGIS
- **ML & Analytics**: Scikit-learn, Pandas, NumPy, H3
- **Geospatial**: GeoPandas, Shapely, Folium
- **Frontend**: Streamlit, Plotly, Kepler.gl
- **Infrastructure**: Docker, Docker Compose

## Let's Connect! 一緒に学びましょう 🌐

[![Twitter Badge](https://img.shields.io/badge/-@GastonEchenique-1DA1F2?style=flat&logo=x&logoColor=white&link=https://x.com/GastonEchenique)](https://x.com/GastonEchenique)
[![LinkedIn Badge](https://img.shields.io/badge/-Gastón_Echenique-0A66C2?style=flat&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/gaston-echenique/)](https://www.linkedin.com/in/gaston-echenique/)
[![GitHub Badge](https://img.shields.io/badge/-oechenique-333?style=flat&logo=github&logoColor=white&link=https://github.com/oechenique)](https://github.com/oechenique)
[![GeoAnalytics Badge](https://img.shields.io/badge/-GeoAnalytics_Site-2ecc71?style=flat&logo=google-earth&logoColor=white&link=https://oechenique.github.io/geoanalytics/)](https://oechenique.github.io/geoanalytics/)
[![Discord Badge](https://img.shields.io/badge/-Gastón|ガストン-5865F2?style=flat&logo=discord&logoColor=white&link=https://discord.com/users/gastonechenique)](https://discord.com/users/gastonechenique)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-FFDD00?style=flat&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/rhrqmdyaig)

## 📝 License | ライセンス

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments | 謝辞

- Data provided by public bike-sharing datasets
- Inspired by urban mobility research
- Built with open-source technologies

---

*🚲 このプロジェクトは単なるETLではなく、本格的な地理空間分析と機械学習インフラです。*  
*Built with ❤️ for ML engineers and urban mobility enthusiasts* | *データサイエンティストと都市モビリティ愛好家のために作られました*