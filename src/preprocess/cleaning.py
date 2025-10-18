"""Módulo para limpieza y validación de datos de transacciones."""

import logging

import pandas as pd

# Configurar logging
logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:  # noqa: C901
    """Limpieza básica de los datos para predicción de Discount Applied."""
    df_clean = df.copy()
    logger.info("Iniciando limpieza de datos...")

    # 1. Manejo de valores nulos
    rows_with_nulls_before = df_clean.isnull().any(axis=1).sum()
    if rows_with_nulls_before > 0:
        logger.info(f"Registros con nulos: {rows_with_nulls_before}")

    # Para Discount Applied, llenar False donde sea nulo
    if "Discount Applied" in df_clean.columns:
        df_clean["Discount Applied"] = df_clean["Discount Applied"].fillna(
            False
        )

    # Para Quantity, llenar con la mediana
    if "Quantity" in df_clean.columns:
        df_clean["Quantity"] = df_clean["Quantity"].fillna(
            df_clean["Quantity"].median()
        )

    # Para Total Spent, llenar con la mediana
    if "Total Spent" in df_clean.columns:
        df_clean["Total Spent"] = df_clean["Total Spent"].fillna(
            df_clean["Total Spent"].median()
        )

    # Para Price Per Unit, calcular desde Total Spent / Quantity
    if "Price Per Unit" in df_clean.columns:
        mask1 = df_clean["Price Per Unit"].isna()
        df_clean.loc[mask1, "Price Per Unit"] = (
            df_clean.loc[mask1, "Total Spent"]
            / df_clean.loc[mask1, "Quantity"]
        )

    # Para Item, imputar usando la moda
    if "Item" in df_clean.columns:
        mask_item = df_clean["Item"].isna()
        df_clean.loc[mask_item, "Item"] = df_clean.groupby(
            ["Category", "Price Per Unit"]
        )["Item"].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA
        )

    rows_with_nulls_after = df_clean.isnull().any(axis=1).sum()
    if rows_with_nulls_after > 0:
        msg = (
            f"Registros con nulos después de imputación: "
            f"{rows_with_nulls_after}"
        )
        logger.warning(msg)

    # 2. Eliminar duplicados
    duplicates = df_clean.duplicated(subset=["Transaction ID"]).sum()
    if duplicates > 0:
        logger.info(f"Duplicados eliminados: {duplicates}")
    df_clean = df_clean.drop_duplicates(subset=["Transaction ID"])

    # 3. Convertir fecha a datetime
    if "Transaction Date" in df_clean.columns:
        df_clean["Transaction Date"] = pd.to_datetime(
            df_clean["Transaction Date"], errors="coerce"
        )

    # 4. Validar consistencia: Total Spent ≈ Price Per Unit * Quantity
    if {"Total Spent", "Price Per Unit", "Quantity"}.issubset(
        df_clean.columns
    ):
        calculated_total = df_clean["Price Per Unit"] * df_clean["Quantity"]
        discrepancy = abs(df_clean["Total Spent"] - calculated_total)
        mask_inconsistent = discrepancy > (df_clean["Total Spent"] * 0.01)
        inconsistent_count = mask_inconsistent.sum()
        if inconsistent_count > 0:
            msg = (
                f"Registros con inconsistencias corregidos: "
                f"{inconsistent_count}"
            )
            logger.info(msg)
        df_clean.loc[mask_inconsistent, "Total Spent"] = calculated_total[
            mask_inconsistent
        ]

    logger.info(f"Limpieza completada: {df_clean.shape}")
    return df_clean
