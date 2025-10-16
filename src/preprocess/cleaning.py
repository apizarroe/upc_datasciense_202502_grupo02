"""Módulo para limpieza y validación de datos de transacciones."""

import pandas as pd


def clean_data(df):
    """Limpieza básica de los datos para predicción de Total Spent."""
    df_clean = df.copy()

    print("Iniciando limpieza de datos...")

    # 1. Manejo de valores nulos
    print("Manejo de valores nulos...")

    # Contar registros con nulos antes de la imputación
    rows_with_nulls_before = df_clean.isnull().any(axis=1).sum()
    print(f"Registros con nulos antes de imputación: {rows_with_nulls_before}")

    # Para Discount Applied, llenar False donde sea nulo
    # df_clean["Discount Applied"] = df_clean["Discount Applied"].fillna(False)

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

    # Para Item, imputar usando la moda por Category y Price Per Unit
    if "Item" in df_clean.columns:
        mask_item = df_clean["Item"].isna()
        df_clean.loc[mask_item, "Item"] = df_clean.groupby(
            ["Category", "Price Per Unit"]
        )["Item"].transform(
            lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA
        )

    # Contar registros con nulos después de la imputación
    rows_with_nulls_after = df_clean.isnull().any(axis=1).sum()
    print(
        f"Registros con nulos después de imputación: {rows_with_nulls_after}"
    )

    # 2. Eliminar duplicados
    print("Eliminando duplicados...")
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["Transaction ID"])
    final_rows = len(df_clean)
    print(f"Duplicados eliminados: {initial_rows - final_rows}")

    # 3. Validar tipos de datos
    print("Validando tipos de datos...")

    # Convertir fecha a datetime
    if "Transaction Date" in df_clean.columns:
        df_clean["Transaction Date"] = pd.to_datetime(
            df_clean["Transaction Date"], errors="coerce"
        )

    # 4. Validar consistencia: Total Spent ≈ Price Per Unit * Quantity
    print("Validando consistencia de datos...")
    calculated_total = df_clean["Price Per Unit"] * df_clean["Quantity"]
    discrepancy = abs(df_clean["Total Spent"] - calculated_total)

    # Corregir valores inconsistentes
    mask_inconsistent = discrepancy > (
        df_clean["Total Spent"] * 0.01
    )  # 1% de tolerancia
    df_clean.loc[mask_inconsistent, "Total Spent"] = calculated_total[
        mask_inconsistent
    ]

    print(
        f"Registros imputados: {rows_with_nulls_before - rows_with_nulls_after}"
    )
    print(f"Inconsistencias corregidas: {mask_inconsistent.sum()}")

    print(f"Dataset después de limpieza: {df_clean.shape}")
    return df_clean
