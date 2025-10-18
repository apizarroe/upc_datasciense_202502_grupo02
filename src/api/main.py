"""API REST para prediccion de descuentos usando FastAPI."""

import logging
import os
import sys
from typing import Dict

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Agregar path del proyecto para imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
)
from src.pipeline.predict import (
    load_model_artifacts,
    predict_batch,
    predict_discount_applied,
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Crear aplicacion FastAPI
app = FastAPI(
    title="Discount Prediction API",
    description="API para predecir si se aplicara descuento a una transaccion",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En produccion, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variables globales para cachear modelo
MODEL_ARTIFACTS: Dict = {}
MODEL_DIR = "data/models"


# =============================================================================
# EVENTOS DE CICLO DE VIDA
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Cargar modelo al iniciar la aplicacion."""
    global MODEL_ARTIFACTS
    try:
        logger.info("Iniciando API...")
        logger.info(f"Cargando modelo desde: {MODEL_DIR}")
        MODEL_ARTIFACTS = load_model_artifacts(model_dir=MODEL_DIR)
        logger.info("✅ Modelo cargado exitosamente")
        logger.info(f"   Tipo: {type(MODEL_ARTIFACTS['model']).__name__}")
        logger.info(f"   Features: {len(MODEL_ARTIFACTS['feature_columns'])}")
    except Exception as e:
        logger.error(f"❌ Error al cargar modelo: {e}")
        logger.error("La API iniciara pero las predicciones fallaran")
        MODEL_ARTIFACTS = {}


@app.on_event("shutdown")
async def shutdown_event():
    """Limpiar recursos al cerrar la aplicacion."""
    logger.info("Cerrando API...")


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.get("/", tags=["General"])
async def root():
    """Endpoint raiz."""
    return {
        "message": "Discount Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    model_loaded = bool(MODEL_ARTIFACTS)

    model_info = None
    if model_loaded:
        model = MODEL_ARTIFACTS["model"]
        model_info = ModelInfo(
            model_type=type(model).__name__,
            n_features=len(MODEL_ARTIFACTS["feature_columns"]),
            model_depth=model.get_depth() if hasattr(model, "get_depth") else None,
            n_leaves=model.get_n_leaves() if hasattr(model, "get_n_leaves") else None,
        )

    return HealthResponse(
        status="healthy" if model_loaded else "model_not_loaded",
        model_loaded=model_loaded,
        model_info=model_info,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
)
async def predict_single(request: PredictionRequest):
    """Predecir si se aplicara descuento a una transaccion individual.

    Args
    ----
        request: PredictionRequest con features preprocesadas

    Returns
    -------
        PredictionResponse con prediccion y probabilidades

    Raises
    ------
        HTTPException 503: Si el modelo no esta cargado
        HTTPException 400: Si hay error en los datos de entrada
        HTTPException 500: Si hay error interno en la prediccion

    Ejemplo de request
    ------------------
        {
            "features": {
                "Age": 0.5234,
                "Quantity": 1.234,
                "Price Per Unit": -0.456,
                "Category_0": 1,
                "Category_1": 0,
                "Gender_Male": 1,
                ...
            }
        }

    """
    # Verificar que el modelo este cargado
    if not MODEL_ARTIFACTS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Ejecute el entrenamiento primero.",
        )

    try:
        logger.info("📥 Recibiendo request de prediccion individual")

        # Realizar prediccion
        result = predict_discount_applied(
            data=request.features, model_dir=MODEL_DIR
        )

        logger.info(
            f"✅ Prediccion exitosa: {result['discount_applied']} "
            f"(confianza: {result['confidence']:.2%})"
        )

        return PredictionResponse(**result)

    except ValueError as e:
        logger.error(f"❌ Error de validacion: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error en datos de entrada: {str(e)}",
        )
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
)
async def predict_batch_endpoint(request: BatchPredictionRequest):
    """Predecir descuentos para multiples transacciones en batch.

    Args
    ----
        request: BatchPredictionRequest con lista de features

    Returns
    -------
        BatchPredictionResponse con predicciones

    Raises
    ------
        HTTPException 503: Si el modelo no esta cargado
        HTTPException 400: Si hay error en los datos de entrada
        HTTPException 500: Si hay error interno en la prediccion

    Ejemplo de request
    ------------------
        {
            "data_list": [
                {"Age": 0.5234, "Quantity": 1.234, ...},
                {"Age": -0.234, "Quantity": 2.567, ...}
            ]
        }

    """
    # Verificar que el modelo este cargado
    if not MODEL_ARTIFACTS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible. Ejecute el entrenamiento primero.",
        )

    try:
        logger.info(f"📥 Recibiendo request de prediccion batch: {len(request.data_list)} items")

        # Realizar prediccion batch
        results = predict_batch(
            data_list=request.data_list, model_dir=MODEL_DIR, save_results=False
        )

        # Calcular estadisticas
        discount_count = sum(1 for r in results if r["discount_applied"])
        no_discount_count = len(results) - discount_count

        logger.info(
            f"✅ Prediccion batch exitosa: {len(results)} items "
            f"({discount_count} con descuento, {no_discount_count} sin descuento)"
        )

        return BatchPredictionResponse(
            predictions=results,
            total=len(results),
            discount_count=discount_count,
            no_discount_count=no_discount_count,
        )

    except ValueError as e:
        logger.error(f"❌ Error de validacion: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error en datos de entrada: {str(e)}",
        )
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}",
        )


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Obtener informacion del modelo cargado."""
    if not MODEL_ARTIFACTS:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo no disponible",
        )

    model = MODEL_ARTIFACTS["model"]
    feature_columns = MODEL_ARTIFACTS["feature_columns"]

    return {
        "model_type": type(model).__name__,
        "model_path": MODEL_ARTIFACTS["model_path"],
        "n_features": len(feature_columns),
        "feature_columns": feature_columns[:10],  # Primeras 10 features
        "total_feature_columns": len(feature_columns),
        "model_depth": model.get_depth() if hasattr(model, "get_depth") else None,
        "n_leaves": model.get_n_leaves() if hasattr(model, "get_n_leaves") else None,
    }


# =============================================================================
# MANEJO DE ERRORES
# =============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Manejador global de excepciones."""
    logger.error(f"Error no manejado: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Error interno del servidor"},
    )


# =============================================================================
# EJECUTAR SERVIDOR
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("🚀 Iniciando servidor de desarrollo...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
