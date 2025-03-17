from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import time
from typing import Dict

from .database import init_db
from .endpoints import router as api_router
from .models import ErrorResponse

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clase para métricas de la API
class APIMetrics:
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.endpoint_latency: Dict[str, float] = {}
        self.start_time = datetime.utcnow()

    def add_request(self):
        self.request_count += 1

    def add_error(self):
        self.error_count += 1

    def update_latency(self, endpoint: str, latency: float):
        if endpoint not in self.endpoint_latency:
            self.endpoint_latency[endpoint] = latency
        else:
            # Promedio móvil
            self.endpoint_latency[endpoint] = (
                self.endpoint_latency[endpoint] * 0.9 + latency * 0.1
            )

    def get_stats(self):
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        return {
            "uptime_seconds": uptime,
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "error_rate": (self.error_count / self.request_count * 100) if self.request_count > 0 else 0,
            "endpoint_latency": self.endpoint_latency
        }

metrics = APIMetrics()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejador del ciclo de vida de la aplicación."""
    # Inicialización
    logger.info("🚀 Iniciando API...")
    try:
        init_db()
        logger.info("✅ Base de datos inicializada")
    except Exception as e:
        logger.error(f"❌ Error inicializando base de datos: {str(e)}")
        raise
    
    yield
    
    # Limpieza
    logger.info("👋 Cerrando API...")

# Crear aplicación FastAPI
app = FastAPI(
    title="Geospatial ML API",
    description="""
    API para análisis geoespacial y machine learning de oportunidades comerciales.
    Proporciona endpoints para consultar datos H3, clusters y predicciones.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware para métricas y logging
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware para recolectar métricas y manejar errores."""
    start_time = time.time()
    path = request.url.path
    
    try:
        # Procesar request
        metrics.add_request()
        response = await call_next(request)
        
        # Actualizar métricas
        latency = time.time() - start_time
        metrics.update_latency(path, latency)
        
        return response
        
    except Exception as e:
        # Manejar errores
        metrics.add_error()
        logger.error(f"Error en {path}: {str(e)}")
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail=str(e),
                error_code="INTERNAL_ERROR"
            ).dict()
        )

# Registrar router principal
app.include_router(
    api_router,
    prefix="/api/v1",
    tags=["geospatial"]
)

# Endpoint de métricas
@app.get("/metrics", tags=["monitoring"])
async def get_metrics():
    """Obtiene métricas de la API."""
    return metrics.get_stats()

# Endpoint de documentación personalizada
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Geospatial ML API",
        version="1.0.0",
        description="""
        API para análisis geoespacial y machine learning de oportunidades comerciales.
        
        Funcionalidades principales:
        * Consulta de datos H3
        * Análisis de clusters
        * Predicciones de potencial comercial
        * Estadísticas y métricas
        * Exportación de datos
        
        Para más información, visita la documentación completa.
        """,
        routes=app.routes,
    )
    
    # Personalizar schema
    openapi_schema["info"]["x-logo"] = {
        "url": "https://your-logo-url.com/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Manejadores de errores
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Manejador de excepciones HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=exc.detail,
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Manejador de excepciones generales."""
    logger.error(f"Error no manejado: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Error interno del servidor",
            error_code="INTERNAL_ERROR"
        ).dict()
    )

# Endpoints informativos
@app.get("/", tags=["info"])
async def root():
    """Endpoint raíz con información básica."""
    return {
        "name": "Geospatial ML API",
        "version": "1.0.0",
        "status": "active",
        "docs_url": "/docs",
        "metrics_url": "/metrics"
    }

@app.get("/status", tags=["info"])
async def status():
    """Estado detallado del servicio."""
    return {
        "service": "Geospatial ML API",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": metrics.get_stats()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )