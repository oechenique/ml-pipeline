import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient
import os
from dotenv import load_dotenv
import h3
from shapely.geometry import Polygon
import json

from src.api.main import app
from src.api.database import Base, get_db

# Cargar variables de entorno de test
load_dotenv(".env.test", override=True)

# Configuración de base de datos de test
DATABASE_URL = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"
)

# Crear engine de test
engine = create_engine(DATABASE_URL)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="session")
def db_engine():
    """Fixture para el engine de base de datos."""
    return engine

@pytest.fixture(scope="function")
def db_session():
    """Fixture para sesiones de base de datos."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db_session):
    """Fixture para el cliente de test de FastAPI."""
    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()
    
    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app)

@pytest.fixture(scope="function")
def sample_h3_data():
    """Fixture para datos de ejemplo H3."""
    # Generar algunos hexágonos H3 de ejemplo
    lat, lng = -34.6037, -58.3816  # Buenos Aires
    h3_index = h3.geo_to_h3(lat, lng, 9)
    
    # Crear geometría del hexágono
    hex_boundary = h3.h3_to_geo_boundary(h3_index)
    polygon = Polygon(hex_boundary)
    
    return {
        "h3_index": h3_index,
        "geometry": json.dumps(polygon.__geo_interface__),
        "population": 1000,
        "poi_count": 50,
        "commercial_density": 0.05,
        "population_density": 10000,
        "cluster": 1,
        "prediction_score": 0.75
    }

@pytest.fixture(scope="function")
def sample_clusters_data():
    """Fixture para datos de ejemplo de clusters."""
    return [
        {
            "cluster_id": 1,
            "statistics": {
                "cell_count": 100,
                "avg_density": 0.05,
                "avg_population": 1000,
                "total_pois": 5000
            }
        },
        {
            "cluster_id": 2,
            "statistics": {
                "cell_count": 150,
                "avg_density": 0.03,
                "avg_population": 800,
                "total_pois": 3000
            }
        }
    ]

@pytest.fixture(scope="function")
def sample_business_types():
    """Fixture para datos de ejemplo de tipos de negocios."""
    return [
        {"type": "restaurant", "count": 100},
        {"type": "retail", "count": 150},
        {"type": "service", "count": 80}
    ]

@pytest.fixture(scope="function")
def auth_headers():
    """Fixture para headers de autenticación (si se implementa)."""
    return {"Authorization": "Bearer test-token"}