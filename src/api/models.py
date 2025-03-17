from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Union
from datetime import datetime
import h3
from shapely.geometry import shape
import json

class BusinessType(BaseModel):
    """Modelo para tipos de negocios."""
    type: str
    count: int = Field(ge=0)
    
    class Config:
        orm_mode = True

class CellMetric(BaseModel):
    """Modelo para métricas de celda."""
    metric_name: str
    metric_value: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        orm_mode = True

class H3CellBase(BaseModel):
    """Modelo base para hexágonos H3."""
    h3_index: str
    population: Optional[int] = Field(ge=0)
    poi_count: Optional[int] = Field(ge=0)
    commercial_density: Optional[float] = Field(ge=0.0)
    population_density: Optional[float] = Field(ge=0.0)
    cluster: Optional[int]
    prediction_score: Optional[float]
    
    @validator('h3_index')
    def validate_h3_index(cls, v):
        """Valida que el índice H3 sea válido."""
        try:
            if not h3.h3_is_valid(v):
                raise ValueError('Índice H3 inválido')
            return v
        except Exception as e:
            raise ValueError(f'Error validando índice H3: {str(e)}')

class H3CellCreate(H3CellBase):
    """Modelo para crear hexágonos H3."""
    geometry: str  # GeoJSON string
    business_types: Optional[List[BusinessType]]
    metrics: Optional[List[CellMetric]]
    
    @validator('geometry')
    def validate_geometry(cls, v):
        """Valida que la geometría sea un GeoJSON válido."""
        try:
            geom = json.loads(v)
            shape(geom)  # Validar geometría con Shapely
            return v
        except Exception as e:
            raise ValueError(f'Geometría GeoJSON inválida: {str(e)}')

class H3CellUpdate(BaseModel):
    """Modelo para actualizar hexágonos H3."""
    population: Optional[int] = Field(ge=0)
    poi_count: Optional[int] = Field(ge=0)
    commercial_density: Optional[float] = Field(ge=0.0)
    population_density: Optional[float] = Field(ge=0.0)
    cluster: Optional[int]
    prediction_score: Optional[float]

class H3CellResponse(H3CellBase):
    """Modelo para respuestas con hexágonos H3."""
    id: int
    geometry: Dict  # GeoJSON
    business_types: List[BusinessType]
    metrics: List[CellMetric]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        orm_mode = True

class BoundingBox(BaseModel):
    """Modelo para bounding boxes."""
    min_lat: float = Field(..., ge=-90, le=90)
    min_lon: float = Field(..., ge=-180, le=180)
    max_lat: float = Field(..., ge=-90, le=90)
    max_lon: float = Field(..., ge=-180, le=180)
    
    @validator('max_lat')
    def validate_lat_bounds(cls, v, values):
        """Valida que max_lat sea mayor que min_lat."""
        if 'min_lat' in values and v <= values['min_lat']:
            raise ValueError('max_lat debe ser mayor que min_lat')
        return v
    
    @validator('max_lon')
    def validate_lon_bounds(cls, v, values):
        """Valida que max_lon sea mayor que min_lon."""
        if 'min_lon' in values and v <= values['min_lon']:
            raise ValueError('max_lon debe ser mayor que min_lon')
        return v

class ClusterStatistics(BaseModel):
    """Modelo para estadísticas de cluster."""
    cell_count: int
    avg_density: float
    avg_population: float
    total_pois: int
    dominant_business_types: List[BusinessType]

class ClusterResponse(BaseModel):
    """Modelo para respuestas de cluster."""
    cluster_id: int
    statistics: ClusterStatistics
    bounds: BoundingBox
    created_at: datetime
    
    class Config:
        orm_mode = True

class PredictionRequest(BaseModel):
    """Modelo para solicitudes de predicción."""
    h3_index: str
    features: Dict[str, float]
    
    @validator('h3_index')
    def validate_h3_index(cls, v):
        if not h3.h3_is_valid(v):
            raise ValueError('Índice H3 inválido')
        return v

class PredictionResponse(BaseModel):
    """Modelo para respuestas de predicción."""
    h3_index: str
    prediction_score: float
    confidence: float
    feature_importance: Dict[str, float]
    similar_areas: List[str]  # Lista de índices H3 similares
    created_at: datetime
    
    class Config:
        orm_mode = True

class BusinessTypeResponse(BaseModel):
    """Modelo para respuestas de tipos de negocio."""
    type: str
    total: int
    percentage: float
    growth_rate: Optional[float]
    
    class Config:
        orm_mode = True

class StatisticsResponse(BaseModel):
    """Modelo para respuestas de estadísticas."""
    total_cells: int
    total_population: int
    total_pois: int
    avg_density: float
    clusters_count: int
    timestamp: datetime
    region_statistics: Optional[Dict[str, Dict[str, Union[int, float]]]]
    
    class Config:
        orm_mode = True

class ErrorResponse(BaseModel):
    """Modelo para respuestas de error."""
    detail: str
    error_code: Optional[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Modelos para paginación
class PaginationParams(BaseModel):
    """Parámetros de paginación."""
    page: int = Field(1, ge=1)
    per_page: int = Field(10, ge=1, le=100)

class PaginatedResponse(BaseModel):
    """Respuesta paginada genérica."""
    items: List
    total: int
    page: int
    per_page: int
    pages: int

# Modelos para filtros
class DateRange(BaseModel):
    """Rango de fechas para filtros."""
    start_date: datetime
    end_date: datetime
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        if 'start_date' in values and v <= values['start_date']:
            raise ValueError('end_date debe ser posterior a start_date')
        return v

class FilterParams(BaseModel):
    """Parámetros de filtrado."""
    date_range: Optional[DateRange]
    min_population: Optional[int] = Field(None, ge=0)
    min_density: Optional[float] = Field(None, ge=0.0)
    business_types: Optional[List[str]]
    clusters: Optional[List[int]]
    
    class Config:
        extra = 'forbid'  # No permitir campos adicionales

# Modelos para actualizaciones en lote
class BulkUpdateItem(BaseModel):
    """Item para actualización en lote."""
    h3_index: str
    updates: H3CellUpdate

class BulkUpdateRequest(BaseModel):
    """Solicitud de actualización en lote."""
    items: List[BulkUpdateItem]
    
    @validator('items')
    def validate_items_limit(cls, v):
        if len(v) > 1000:  # Límite arbitrario
            raise ValueError('Demasiados items para actualización en lote')
        return v

class BulkUpdateResponse(BaseModel):
    """Respuesta de actualización en lote."""
    successful: int
    failed: int
    errors: List[Dict[str, str]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)