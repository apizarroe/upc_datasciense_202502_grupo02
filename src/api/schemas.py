"""Esquemas de validacion para la API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request para prediccion individual."""

    features: Dict[str, float] = Field(
        ...,
        description="Diccionario con features preprocesadas",
        example={
            "Age": 0.5234,
            "Quantity": 1.234,
            "Price Per Unit": -0.456,
            "Category_0": 1,
            "Category_1": 0,
            "Gender_Male": 1,
        },
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "Age": 0.5234,
                    "Quantity": 1.234,
                    "Price Per Unit": -0.456,
                    "Category_0": 1,
                    "Category_1": 0,
                    "Gender_Male": 1,
                }
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request para prediccion en batch."""

    data_list: List[Dict[str, float]] = Field(
        ...,
        description="Lista de diccionarios con features preprocesadas",
        min_length=1,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "data_list": [
                    {
                        "Age": 0.5234,
                        "Quantity": 1.234,
                        "Price Per Unit": -0.456,
                        "Category_0": 1,
                    },
                    {
                        "Age": -0.234,
                        "Quantity": 2.567,
                        "Price Per Unit": 0.123,
                        "Category_0": 0,
                    },
                ]
            }
        }


class ModelInfo(BaseModel):
    """Informacion del modelo."""

    model_type: str
    n_features: int
    model_depth: Optional[int] = None
    n_leaves: Optional[int] = None


class PredictionResponse(BaseModel):
    """Respuesta de prediccion individual."""

    discount_applied: bool = Field(..., description="Si se aplica descuento")
    prediction: int = Field(..., description="Prediccion (0 o 1)")
    probability_no_discount: float = Field(
        ..., description="Probabilidad de no descuento"
    )
    probability_discount: float = Field(..., description="Probabilidad de descuento")
    confidence: float = Field(..., description="Confianza de la prediccion")
    model_info: ModelInfo
    timestamp: str


class BatchPredictionItem(BaseModel):
    """Item de prediccion en batch."""

    index: int
    discount_applied: bool
    prediction: int
    probability_no_discount: float
    probability_discount: float
    confidence: float


class BatchPredictionResponse(BaseModel):
    """Respuesta de prediccion en batch."""

    predictions: List[BatchPredictionItem]
    total: int
    discount_count: int
    no_discount_count: int


class HealthResponse(BaseModel):
    """Respuesta de health check."""

    status: str
    model_loaded: bool
    model_info: Optional[ModelInfo] = None
