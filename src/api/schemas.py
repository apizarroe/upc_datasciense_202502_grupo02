"""Esquemas de validacion para la API."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request para prediccion individual."""

    features: Dict[str, float] = Field(
        ...,
        description="Diccionario con las 28 features preprocesadas del modelo",
        example={
            "Price Per Unit": -0.453,
            "Quantity": 1.592,
            "Customer_Transaction_Count": 0.726,
            "Customer_Avg_Spent": -1.755,
            "Customer_Discount_Frequency": -0.292,
            "Customer_Recency": -0.267,
            "Product_Transaction_Count": -1.642,
            "Product_Avg_Price": -0.453,
            "Product_Discount_Rate": -0.086,
            "Product_Popularity_Score": -1.642,
            "Transaction_Year": 2024,
            "Transaction_Month": 4,
            "Transaction_Quarter": 2,
            "Transaction_DayOfWeek": 0,
            "Price_Quantity_Interaction": 0.586,
            "Category_0": 0,
            "Category_1": 0,
            "Category_2": 0,
            "Category_3": 0,
            "Category_4": 0,
            "Category_5": 0,
            "Category_6": 0,
            "Category_7": 1,
            "Location_0": 0,
            "Location_1": 1,
            "Payment Method_0": 0,
            "Payment Method_1": 0,
            "Payment Method_2": 1,
        },
    )

    class Config:
        json_schema_extra = {
            "example": {
                "features": {
                    "Price Per Unit": -0.453,
                    "Quantity": 1.592,
                    "Customer_Transaction_Count": 0.726,
                    "Customer_Avg_Spent": -1.755,
                    "Customer_Discount_Frequency": -0.292,
                    "Customer_Recency": -0.267,
                    "Product_Transaction_Count": -1.642,
                    "Product_Avg_Price": -0.453,
                    "Product_Discount_Rate": -0.086,
                    "Product_Popularity_Score": -1.642,
                    "Transaction_Year": 2024,
                    "Transaction_Month": 4,
                    "Transaction_Quarter": 2,
                    "Transaction_DayOfWeek": 0,
                    "Price_Quantity_Interaction": 0.586,
                    "Category_0": 0,
                    "Category_1": 0,
                    "Category_2": 0,
                    "Category_3": 0,
                    "Category_4": 0,
                    "Category_5": 0,
                    "Category_6": 0,
                    "Category_7": 1,
                    "Location_0": 0,
                    "Location_1": 1,
                    "Payment Method_0": 0,
                    "Payment Method_1": 0,
                    "Payment Method_2": 1,
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
                        "Price Per Unit": -0.453,
                        "Quantity": 1.592,
                        "Customer_Transaction_Count": 0.726,
                        "Customer_Avg_Spent": -1.755,
                        "Customer_Discount_Frequency": -0.292,
                        "Customer_Recency": -0.267,
                        "Product_Transaction_Count": -1.642,
                        "Product_Avg_Price": -0.453,
                        "Product_Discount_Rate": -0.086,
                        "Product_Popularity_Score": -1.642,
                        "Transaction_Year": 2024,
                        "Transaction_Month": 4,
                        "Transaction_Quarter": 2,
                        "Transaction_DayOfWeek": 0,
                        "Price_Quantity_Interaction": 0.586,
                        "Category_0": 0,
                        "Category_1": 0,
                        "Category_2": 0,
                        "Category_3": 0,
                        "Category_4": 0,
                        "Category_5": 0,
                        "Category_6": 0,
                        "Category_7": 1,
                        "Location_0": 0,
                        "Location_1": 1,
                        "Payment Method_0": 0,
                        "Payment Method_1": 0,
                        "Payment Method_2": 1,
                    },
                    {
                        "Price Per Unit": 0.234,
                        "Quantity": -0.567,
                        "Customer_Transaction_Count": -1.234,
                        "Customer_Avg_Spent": 0.891,
                        "Customer_Discount_Frequency": 0.456,
                        "Customer_Recency": 0.123,
                        "Product_Transaction_Count": 0.789,
                        "Product_Avg_Price": 0.234,
                        "Product_Discount_Rate": 0.345,
                        "Product_Popularity_Score": 0.789,
                        "Transaction_Year": 2023,
                        "Transaction_Month": 8,
                        "Transaction_Quarter": 3,
                        "Transaction_DayOfWeek": 2,
                        "Price_Quantity_Interaction": -0.234,
                        "Category_0": 1,
                        "Category_1": 0,
                        "Category_2": 0,
                        "Category_3": 0,
                        "Category_4": 0,
                        "Category_5": 0,
                        "Category_6": 0,
                        "Category_7": 0,
                        "Location_0": 1,
                        "Location_1": 0,
                        "Payment Method_0": 1,
                        "Payment Method_1": 0,
                        "Payment Method_2": 0,
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
