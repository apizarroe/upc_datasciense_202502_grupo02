"""Módulo para crear características agregadas OPTIMIZADAS de clientes y productos."""

import pandas as pd


def create_customer_features(df):
    """Crear características agregadas por cliente OPTIMIZADAS."""
    print("Creando características de cliente OPTIMIZADAS...")

    # SOLO características realmente útiles para predecir Total Spent
    customer_features = (
        df.groupby("Customer ID")
        .agg({
            "Transaction ID": "count",           # Frecuencia de compra
            "Total Spent": "mean",               # Gasto promedio
            "Discount Applied": "mean",          # Frecuencia de descuentos
        })
        .round(2)
    )

    # Renombrar columnas
    customer_features.columns = [
        "Customer_Transaction_Count",
        "Customer_Avg_Spent",
        "Customer_Discount_Frequency"
    ]

    # Solo una métrica adicional importante
    customer_features["Customer_Recency"] = (
        pd.to_datetime("today") - df.groupby("Customer ID")["Transaction Date"].max()
    ).dt.days

    return customer_features.reset_index()


def create_product_features(df):
    """Crear características agregadas por producto/categoría OPTIMIZADAS."""
    print("Creando características de producto OPTIMIZADAS...")

    # SOLO características relevantes para predecir Total Spent
    product_features = (
        df.groupby(["Category", "Item"])
        .agg({
            "Transaction ID": "count",           # Popularidad
            "Price Per Unit": "mean",            # Precio promedio
            "Discount Applied": "mean",          # Frecuencia de descuentos
        })
        .round(2)
    )

    # Renombrar columnas
    product_features.columns = [
        "Product_Transaction_Count",
        "Product_Avg_Price",
        "Product_Discount_Rate"
    ]

    # Solo una métrica adicional importante
    total_transactions = product_features["Product_Transaction_Count"].sum()
    product_features["Product_Popularity_Score"] = (
        product_features["Product_Transaction_Count"] / total_transactions
    )

    return product_features.reset_index()


def enrich_transaction_data(df, customer_features, product_features):
    """Enriquecer datos de transacción con características agregadas OPTIMIZADAS."""
    print("Enriqueciendo datos de transacción OPTIMIZADO...")

    df_enriched = df.copy()

    # Unir características de cliente
    df_enriched = df_enriched.merge(
        customer_features,
        on="Customer ID",
        how="left"
    )

    # Unir características de producto
    df_enriched = df_enriched.merge(
        product_features,
        on=["Category", "Item"],
        how="left"
    )

    print(f"Dataset enriquecido: {df_enriched.shape}")
    return df_enriched
