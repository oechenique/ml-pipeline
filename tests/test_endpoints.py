import pytest
from fastapi import status
import h3

def test_health_check(client):
    """Test del endpoint de health check."""
    response = client.get("/api/v1/health")
    assert response.status_code == status.HTTP_200_OK
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_get_cell(client, db_session, sample_h3_data):
    """Test de obtención de un hexágono H3."""
    # Crear un hexágono de prueba
    h3_index = sample_h3_data["h3_index"]
    response = client.post("/api/v1/cells", json=sample_h3_data)
    assert response.status_code == status.HTTP_200_OK
    
    # Obtener el hexágono creado
    response = client.get(f"/api/v1/cells/{h3_index}")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["h3_index"] == h3_index

def test_get_cells_in_area(client, db_session, sample_h3_data):
    """Test de obtención de hexágonos en un área."""
    # Crear algunos hexágonos de prueba
    client.post("/api/v1/cells", json=sample_h3_data)
    
    # Consultar área
    params = {
        "min_lat": -35.0,
        "min_lon": -59.0,
        "max_lat": -34.0,
        "max_lon": -58.0
    }
    response = client.get("/api/v1/cells", params=params)
    assert response.status_code == status.HTTP_200_OK
    assert isinstance(response.json(), list)

def test_get_clusters(client, db_session, sample_clusters_data):
    """Test de obtención de clusters."""
    response = client.get("/api/v1/clusters")
    assert response.status_code == status.HTTP_200_OK
    assert isinstance(response.json(), list)

def test_get_predictions(client, db_session, sample_h3_data):
    """Test de obtención de predicciones."""
    # Crear datos de prueba
    client.post("/api/v1/cells", json=sample_h3_data)
    
    # Obtener predicciones
    response = client.get("/api/v1/predictions/top?limit=5")
    assert response.status_code == status.HTTP_200_OK
    assert isinstance(response.json(), list)
    assert len(response.json()) <= 5

def test_create_cell_validation(client):
    """Test de validación en creación de hexágonos."""
    # Datos inválidos
    invalid_data = {
        "h3_index": "invalid_index",
        "population": -1  # Población negativa
    }
    response = client.post("/api/v1/cells", json=invalid_data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_business_types(client, db_session, sample_business_types):
    """Test de endpoint de tipos de negocios."""
    response = client.get("/api/v1/business-types")
    assert response.status_code == status.HTTP_200_OK
    assert isinstance(response.json(), list)

def test_statistics(client, db_session, sample_h3_data):
    """Test de endpoint de estadísticas."""
    # Crear datos de prueba
    client.post("/api/v1/cells", json=sample_h3_data)
    
    response = client.get("/api/v1/statistics")
    assert response.status_code == status.HTTP_200_OK
    assert "total_cells" in response.json()
    assert "total_population" in response.json()

def test_export_data(client, db_session, sample_h3_data):
    """Test de exportación de datos."""
    # Crear datos de prueba
    client.post("/api/v1/cells", json=sample_h3_data)
    
    # Probar exportación JSON
    response = client.get("/api/v1/export?format=json")
    assert response.status_code == status.HTTP_200_OK
    assert "data" in response.json()
    
    # Probar exportación CSV
    response = client.get("/api/v1/export?format=csv")
    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"] == "text/csv"

@pytest.mark.parametrize("invalid_bbox", [
    {"min_lat": 91},  # Latitud inválida
    {"min_lon": 181},  # Longitud inválida
    {"max_lat": -91},
    {"max_lon": -181}
])
def test_invalid_bounding_box(client, invalid_bbox):
    """Test de validación de bounding box."""
    response = client.get("/api/v1/cells", params=invalid_bbox)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_bulk_operations(client, db_session):
    """Test de operaciones en lote."""
    # Crear múltiples hexágonos
    bulk_data = [
        {
            "h3_index": h3.geo_to_h3(-34.6037, -58.3816, 9),
            "population": 1000,
            "poi_count": 50
        },
        {
            "h3_index": h3.geo_to_h3(-34.6038, -58.3817, 9),
            "population": 1500,
            "poi_count": 75
        }
    ]
    
    response = client.post("/api/v1/cells/bulk", json=bulk_data)
    assert response.status_code == status.HTTP_200_OK
    assert "message" in response.json()

def test_search(client, db_session, sample_h3_data):
    """Test de funcionalidad de búsqueda."""
    # Crear datos de prueba
    client.post("/api/v1/cells", json=sample_h3_data)
    
    # Buscar por varios criterios
    params = {"query": "rest", "limit": 5}  # Buscar restaurantes
    response = client.get("/api/v1/search", params=params)
    assert response.status_code == status.HTTP_200_OK
    assert isinstance(response.json(), list)

def test_metrics(client):
    """Test del endpoint de métricas."""
    response = client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    metrics = response.json()
    assert "uptime_seconds" in metrics
    assert "total_requests" in metrics
    assert "error_count" in metrics
    assert "endpoint_latency" in metrics

def test_error_handling(client):
    """Test del manejo de errores."""
    # Probar 404
    response = client.get("/api/v1/cells/non_existent")
    assert response.status_code == status.HTTP_404_NOT_FOUND
    
    # Probar validación de datos
    response = client.post("/api/v1/cells", json={})
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_pagination(client, db_session, sample_h3_data):
    """Test de paginación."""
    # Crear varios registros
    for i in range(15):
        modified_data = sample_h3_data.copy()
        modified_data["h3_index"] = h3.geo_to_h3(-34.6037 + i*0.01, -58.3816, 9)
        client.post("/api/v1/cells", json=modified_data)
    
    # Probar primera página
    response = client.get("/api/v1/cells?page=1&per_page=10")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["items"]) == 10
    assert data["total"] > 10
    assert data["page"] == 1

def test_filtering(client, db_session, sample_h3_data):
    """Test de filtrado."""
    # Crear dato de prueba
    client.post("/api/v1/cells", json=sample_h3_data)
    
    # Probar filtros
    response = client.get("/api/v1/cells", params={
        "min_population": 500,
        "min_density": 0.01,
        "business_types": ["restaurant"]
    })
    assert response.status_code == status.HTTP_200_OK
    
def test_date_range_filter(client, db_session, sample_h3_data):
    """Test de filtrado por rango de fechas."""
    response = client.get("/api/v1/cells", params={
        "start_date": "2024-01-01",
        "end_date": "2024-12-31"
    })
    assert response.status_code == status.HTTP_200_OK

def test_documentation(client):
    """Test de disponibilidad de documentación."""
    response = client.get("/docs")
    assert response.status_code == status.HTTP_200_OK
    
    response = client.get("/redoc")
    assert response.status_code == status.HTTP_200_OK

def test_openapi_schema(client):
    """Test del schema OpenAPI."""
    response = client.get("/openapi.json")
    assert response.status_code == status.HTTP_200_OK
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema