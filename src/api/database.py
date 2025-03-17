from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from geoalchemy2 import Geometry
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import Generator
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración de la base de datos
DATABASE_URL = f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DB')}"

# Crear engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10
)

# Sesión
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base declarativa
Base = declarative_base()

class H3Cell(Base):
    """Modelo para hexágonos H3."""
    __tablename__ = "h3_cells"

    id = Column(Integer, primary_key=True, index=True)
    h3_index = Column(String, unique=True, index=True, nullable=False)
    geometry = Column(Geometry('POLYGON', srid=4326), nullable=False)
    population = Column(Integer)
    poi_count = Column(Integer)
    commercial_density = Column(Float)
    population_density = Column(Float)
    cluster = Column(Integer)
    prediction_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relaciones
    business_types = relationship("BusinessType", back_populates="h3_cell")
    metrics = relationship("CellMetric", back_populates="h3_cell")

class BusinessType(Base):
    """Modelo para tipos de negocios en cada hexágono."""
    __tablename__ = "business_types"

    id = Column(Integer, primary_key=True, index=True)
    h3_cell_id = Column(Integer, ForeignKey("h3_cells.id"))
    type = Column(String, nullable=False)
    count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relaciones
    h3_cell = relationship("H3Cell", back_populates="business_types")

class CellMetric(Base):
    """Modelo para métricas adicionales de hexágonos."""
    __tablename__ = "cell_metrics"

    id = Column(Integer, primary_key=True, index=True)
    h3_cell_id = Column(Integer, ForeignKey("h3_cells.id"))
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relaciones
    h3_cell = relationship("H3Cell", back_populates="metrics")

def init_db() -> None:
    """Inicializa la base de datos."""
    try:
        # Crear extensión PostGIS si no existe
        with engine.connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")
            conn.execute("CREATE EXTENSION IF NOT EXISTS h3") # Extensión H3 si está disponible
        
        # Crear tablas
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Base de datos inicializada correctamente")
        
    except Exception as e:
        logger.error(f"❌ Error inicializando base de datos: {str(e)}")
        raise

def get_db() -> Generator:
    """
    Dependencia para obtener sesión de base de datos.
    Para usar con FastAPI.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class DatabaseManager:
    """Gestor de operaciones de base de datos."""
    
    def __init__(self):
        self.SessionLocal = SessionLocal
    
    def get_h3_cell(self, db, h3_index: str) -> H3Cell:
        """Obtiene un hexágono H3 por su índice."""
        return db.query(H3Cell).filter(H3Cell.h3_index == h3_index).first()
    
    def get_cells_in_bounds(
        self,
        db,
        min_lat: float,
        min_lon: float,
        max_lat: float,
        max_lon: float
    ) -> list:
        """Obtiene hexágonos dentro de un bounding box."""
        return db.query(H3Cell).filter(
            H3Cell.geometry.ST_Within(
                f'ST_MakeEnvelope({min_lon}, {min_lat}, {max_lon}, {max_lat}, 4326)'
            )
        ).all()
    
    def get_cells_by_cluster(self, db, cluster_id: int) -> list:
        """Obtiene hexágonos de un cluster específico."""
        return db.query(H3Cell).filter(H3Cell.cluster == cluster_id).all()
    
    def get_top_potential_areas(
        self,
        db,
        limit: int = 10,
        min_population: int = 0
    ) -> list:
        """Obtiene áreas con mayor potencial comercial."""
        return db.query(H3Cell).filter(
            H3Cell.population >= min_population
        ).order_by(
            H3Cell.prediction_score.desc()
        ).limit(limit).all()
    
    def get_business_distribution(self, db) -> dict:
        """Obtiene distribución de tipos de negocios."""
        return db.query(
            BusinessType.type,
            func.sum(BusinessType.count).label('total')
        ).group_by(
            BusinessType.type
        ).all()
    
    def get_cluster_statistics(self, db) -> dict:
        """Obtiene estadísticas por cluster."""
        return db.query(
            H3Cell.cluster,
            func.count(H3Cell.id).label('cell_count'),
            func.avg(H3Cell.commercial_density).label('avg_density'),
            func.avg(H3Cell.population).label('avg_population')
        ).group_by(
            H3Cell.cluster
        ).all()
    
    def upsert_h3_cell(
        self,
        db,
        h3_index: str,
        geometry: str,
        data: dict
    ) -> H3Cell:
        """Inserta o actualiza un hexágono H3."""
        cell = self.get_h3_cell(db, h3_index)
        
        if cell:
            for key, value in data.items():
                setattr(cell, key, value)
        else:
            cell = H3Cell(
                h3_index=h3_index,
                geometry=geometry,
                **data
            )
            db.add(cell)
        
        db.commit()
        db.refresh(cell)
        return cell
    
    def bulk_upsert_cells(
        self,
        db,
        cells_data: list
    ) -> None:
        """Inserta o actualiza múltiples hexágonos H3."""
        for cell_data in cells_data:
            self.upsert_h3_cell(
                db,
                cell_data['h3_index'],
                cell_data['geometry'],
                cell_data
            )

# Instancia global del gestor de base de datos
db_manager = DatabaseManager()

# Crear tablas al importar el módulo
init_db()