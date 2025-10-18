"""Módulo para características agregadas de clientes y productos."""

import logging

import pandas as pd

# Configurar logging
logger = logging.getLogger(__name__)


def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crear características agregadas por cliente OPTIMIZADAS."""
    logger.info("Creando características de cliente...")

    # SOLO características realmente útiles para predecir Total Spent
    customer_features = (
        df.groupby("Customer ID")
        .agg(
            {
                "Transaction ID": "count",  # Frecuencia de compra
                "Total Spent": "mean",  # Gasto promedio
                "Discount Applied": "mean",  # Frecuencia de descuentos
            }
        )
        .round(2)
    )

    # Renombrar columnas
    customer_features.columns = pd.Index(
        [
            "Customer_Transaction_Count",
            "Customer_Avg_Spent",
            "Customer_Discount_Frequency",
        ]
    )

    # Solo una métrica adicional importante
    max_dates = df.groupby("Customer ID")["Transaction Date"].max()
    today = pd.Timestamp.now()
    recency = today - max_dates  # type: ignore[operator]
    customer_features["Customer_Recency"] = recency.dt.days

    logger.info(
        f"Características de cliente creadas: {customer_features.shape}"
    )
    return customer_features.reset_index()


def create_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crear características agregadas por producto/categoría OPTIMIZADAS."""
    logger.info("Creando características de producto...")

    # SOLO características relevantes para predecir Total Spent
    product_features = (
        df.groupby(["Category", "Item"])
        .agg(
            {
                "Transaction ID": "count",  # Popularidad
                "Price Per Unit": "mean",  # Precio promedio
                "Discount Applied": "mean",  # Frecuencia de descuentos
            }
        )
        .round(2)
    )

    # Renombrar columnas
    product_features.columns = pd.Index(
        [
            "Product_Transaction_Count",
            "Product_Avg_Price",
            "Product_Discount_Rate",
        ]
    )

    # Solo una métrica adicional importante
    total_transactions = product_features["Product_Transaction_Count"].sum()
    product_features["Product_Popularity_Score"] = (
        product_features["Product_Transaction_Count"] / total_transactions
    )

    logger.info(
        f"Características de producto creadas: {product_features.shape}"
    )
    return product_features.reset_index()


def enrich_transaction_data(
    df: pd.DataFrame,
    customer_features: pd.DataFrame,
    product_features: pd.DataFrame,
) -> pd.DataFrame:
    """Enriquecer datos con características agregadas."""
    logger.info("Enriqueciendo datos de transacción...")

    df_enriched = df.copy()

    # Unir características de cliente
    df_enriched = df_enriched.merge(
        customer_features, on="Customer ID", how="left"
    )

    # Unir características de producto
    df_enriched = df_enriched.merge(
        product_features, on=["Category", "Item"], how="left"
    )

    logger.info(f"Dataset enriquecido: {df_enriched.shape}")
    return df_enriched
