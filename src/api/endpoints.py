from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import json
from .database import get_db, db_manager
from .models import (
    H3CellResponse,
    H3CellCreate,
    ClusterResponse,
    PredictionResponse,
    BusinessTypeResponse,
    StatisticsResponse,
    BoundingBox
)

router = APIRouter()

@router.get("/health")
async def health_check():
    """Endpoint de verificación de salud de la API."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/cells/{h3_index}", response_model=H3CellResponse)
async def get_cell(
    h3_index: str,
    db: Session = Depends(get_db)
):
    """
    Obtiene información detallada de un hexágono H3.
    
    Args:
        h3_index: Índice H3 del hexágono
        db: Sesión de base de datos
    """
    cell = db_manager.get_h3_cell(db, h3_index)
    if not cell:
        raise HTTPException(status_code=404, detail="Hexágono no encontrado")
    return cell

@router.get("/cells", response_model=List[H3CellResponse])
async def get_cells_in_area(
    bounds: BoundingBox = Depends(),
    cluster_id: Optional[int] = None,
    min_population: Optional[int] = None,
    min_density: Optional[float] = None,
    db: Session = Depends(get_db)
):
    """
    Obtiene hexágonos H3 que cumplen con los criterios especificados.
    
    Args:
        bounds: Bounding box para filtrar área
        cluster_id: ID de cluster opcional
        min_population: Población mínima
        min_density: Densidad comercial mínima
        db: Sesión de base de datos
    """
    # Construir query base
    query = db.query(H3Cell)
    
    # Aplicar filtros
    if bounds:
        query = query.filter(
            H3Cell.geometry.ST_Within(
                f'ST_MakeEnvelope({bounds.min_lon}, {bounds.min_lat}, {bounds.max_lon}, {bounds.max_lat}, 4326)'
            )
        )
    
    if cluster_id is not None:
        query = query.filter(H3Cell.cluster == cluster_id)
    
    if min_population:
        query = query.filter(H3Cell.population >= min_population)
    
    if min_density:
        query = query.filter(H3Cell.commercial_density >= min_density)
    
    return query.all()

@router.get("/clusters", response_model=List[ClusterResponse])
async def get_clusters(
    db: Session = Depends(get_db)
):
    """
    Obtiene información de todos los clusters.
    
    Args:
        db: Sesión de base de datos
    """
    stats = db_manager.get_cluster_statistics(db)
    return stats

@router.get("/clusters/{cluster_id}", response_model=ClusterResponse)
async def get_cluster(
    cluster_id: int,
    db: Session = Depends(get_db)
):
    """
    Obtiene información detallada de un cluster específico.
    
    Args:
        cluster_id: ID del cluster
        db: Sesión de base de datos
    """
    cells = db_manager.get_cells_by_cluster(db, cluster_id)
    if not cells:
        raise HTTPException(status_code=404, detail="Cluster no encontrado")
    
    # Calcular estadísticas del cluster
    stats = {
        "cluster_id": cluster_id,
        "cell_count": len(cells),
        "avg_density": sum(c.commercial_density for c in cells) / len(cells),
        "avg_population": sum(c.population for c in cells) / len(cells),
        "total_pois": sum(c.poi_count for c in cells)
    }
    
    return stats

@router.get("/predictions/top", response_model=List[PredictionResponse])
async def get_top_predictions(
    limit: int = Query(10, ge=1, le=100),
    min_population: int = Query(0, ge=0),
    db: Session = Depends(get_db)
):
    """
    Obtiene las áreas con mayor potencial comercial.
    
    Args:
        limit: Número de áreas a retornar
        min_population: Población mínima requerida
        db: Sesión de base de datos
    """
    areas = db_manager.get_top_potential_areas(db, limit, min_population)
    return areas

@router.get("/business-types", response_model=List[BusinessTypeResponse])
async def get_business_types(
    db: Session = Depends(get_db)
):
    """
    Obtiene distribución de tipos de negocios.
    
    Args:
        db: Sesión de base de datos
    """
    distribution = db_manager.get_business_distribution(db)
    return distribution

@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(
    db: Session = Depends(get_db)
):
    """
    Obtiene estadísticas generales del sistema.
    
    Args:
        db: Sesión de base de datos
    """
    # Query para estadísticas generales
    stats = db.query(
        func.count(H3Cell.id).label('total_cells'),
        func.sum(H3Cell.population).label('total_population'),
        func.sum(H3Cell.poi_count).label('total_pois'),
        func.avg(H3Cell.commercial_density).label('avg_density')
    ).first()
    
    return {
        "total_cells": stats.total_cells,
        "total_population": stats.total_population,
        "total_pois": stats.total_pois,
        "avg_density": stats.avg_density,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.post("/cells", response_model=H3CellResponse)
async def create_cell(
    cell: H3CellCreate,
    db: Session = Depends(get_db)
):
    """
    Crea o actualiza un hexágono H3.
    
    Args:
        cell: Datos del hexágono
        db: Sesión de base de datos
    """
    # Validar geometría
    if not cell.geometry:
        raise HTTPException(
            status_code=400,
            detail="Geometría requerida"
        )
    
    try:
        h3_cell = db_manager.upsert_h3_cell(
            db,
            cell.h3_index,
            cell.geometry,
            cell.dict(exclude={'h3_index', 'geometry'})
        )
        return h3_cell
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creando hexágono: {str(e)}"
        )

@router.post("/cells/bulk", response_model=dict)
async def bulk_create_cells(
    cells: List[H3CellCreate],
    db: Session = Depends(get_db)
):
    """
    Crea o actualiza múltiples hexágonos H3.
    
    Args:
        cells: Lista de datos de hexágonos
        db: Sesión de base de datos
    """
    try:
        db_manager.bulk_upsert_cells(db, [cell.dict() for cell in cells])
        return {
            "message": f"Procesados {len(cells)} hexágonos exitosamente",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en proceso bulk: {str(e)}"
        )

@router.get("/search", response_model=List[H3CellResponse])
async def search_cells(
    query: str = Query(..., min_length=3),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Busca hexágonos que coincidan con criterios específicos.
    
    Args:
        query: Texto de búsqueda
        limit: Número máximo de resultados
        db: Sesión de base de datos
    """
    # Implementar lógica de búsqueda
    cells = db.query(H3Cell).filter(
        or_(
            H3Cell.h3_index.ilike(f"%{query}%"),
            BusinessType.type.ilike(f"%{query}%")
        )
    ).join(
        BusinessType,
        isouter=True
    ).limit(limit).all()
    
    return cells

@router.get("/export")
async def export_data(
    format: str = Query("json", regex="^(json|csv)$"),
    bounds: Optional[BoundingBox] = None,
    db: Session = Depends(get_db)
):
    """
    Exporta datos en formato especificado.
    
    Args:
        format: Formato de exportación ('json' o 'csv')
        bounds: Bounding box opcional
        db: Sesión de base de datos
    """
    # Obtener datos
    cells = get_cells_in_area(bounds=bounds, db=db)
    
    if format == "json":
        return {
            "data": [cell.dict() for cell in cells],
            "count": len(cells),
            "timestamp": datetime.utcnow().isoformat()
        }
    else:
        # Convertir a CSV
        import pandas as pd
        df = pd.DataFrame([cell.dict() for cell in cells])
        return Response(
            content=df.to_csv(index=False),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )