"""Módulo para preprocesamiento OPTIMIZADO de datos (clasificación)."""

import logging

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Configurar logging
logger = logging.getLogger(__name__)


def preprocess_for_classification(df: pd.DataFrame, target_column: str = "Discount Applied") -> tuple[pd.DataFrame, dict]:
    """Preprocesamiento para modelo de clasificación (descuento sí/no)."""
    df_processed = df.copy()
    logger.info("Iniciando preprocesamiento para clasificación...")

    # 1. Asegurar que Discount Applied sea numérico (0/1)
    df_processed[target_column] = (
        df_processed[target_column].astype(bool).astype(int)
    )

    # 2. Codificar variables categóricas relevantes
    essential_categoricals = ["Category", "Payment Method", "Location"]
    label_encoders = {}

    for col in essential_categoricals:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le

    # 3. Características temporales
    if "Transaction Date" in df_processed.columns:
        df_processed["Transaction_Year"] = df_processed[
            "Transaction Date"
        ].dt.year
        df_processed["Transaction_Month"] = df_processed[
            "Transaction Date"
        ].dt.month
        df_processed["Transaction_Quarter"] = df_processed[
            "Transaction Date"
        ].dt.quarter
        df_processed["Transaction_DayOfWeek"] = df_processed[
            "Transaction Date"
        ].dt.dayofweek

    # 4. Crear interacciones útiles
    if {"Price Per Unit", "Quantity"}.issubset(df_processed.columns):
        df_processed["Price_Quantity_Interaction"] = (
            df_processed["Price Per Unit"] * df_processed["Quantity"]
        )

    # 5. Eliminar columnas no necesarias o que causan fuga de información
    drop_cols = [
        "Transaction ID",
        "Item",
        "Transaction Date",
        "Total Spent",  # Se elimina porque depende del descuento
    ]
    df_processed = df_processed.drop(
        columns=[c for c in drop_cols if c in df_processed.columns],
        errors="ignore",
    )

    # 6. One-hot encoding para variables categóricas de ubicación o categoría
    categorical_for_onehot = ["Category", "Location", "Payment Method"]
    for col in categorical_for_onehot:
        if col in df_processed.columns:
            dummies = pd.get_dummies(df_processed[col], prefix=col)
            df_processed = pd.concat([df_processed, dummies], axis=1)
            df_processed = df_processed.drop(columns=[col])

    # 7. Escalar solo variables numéricas continuas
    scaler = StandardScaler()
    numeric_cols = df_processed.select_dtypes(
        include=["int64", "float64"]
    ).columns
    numeric_cols = [c for c in numeric_cols if c != target_column]

    df_processed[numeric_cols] = scaler.fit_transform(
        df_processed[numeric_cols]
    )

    # Escalar datos de predicción con el mismo scaler
    # if len(df_for_prediction) > 0:
    #    df_for_prediction[numeric_cols] = scaler.transform(
    #        df_for_prediction[numeric_cols]
    #    )

    logger.info(f"Columnas escaladas: {len(numeric_cols)}")
    logger.info(f"Preprocesamiento completado: {df_processed.shape}")

    return df_processed, label_encoders
