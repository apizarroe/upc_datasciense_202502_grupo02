"""Módulo para limpieza y validación de datos de transacciones."""

import pandas as pd


def clean_data(df):
    """Limpieza básica de los datos para predicción de Total Spent."""
    df_clean = df.copy()

    print("Iniciando limpieza de datos...")

    # 1. Manejo de valores nulos
    print("Manejo de valores nulos...")

    # Para Discount Applied, llenar False donde sea nulo
    df_clean["Discount Applied"] = df_clean["Discount Applied"].fillna(False)

    # Para columnas numéricas, llenar con la mediana
    numeric_columns = ["Price Per Unit", "Quantity", "Total Spent"]
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # 2. Eliminar duplicados
    print("Eliminando duplicados...")
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["Transaction ID"])
    final_rows = len(df_clean)
    print(f"Duplicados eliminados: {initial_rows - final_rows}")

    # 3. Validar tipos de datos
    print("Validando tipos de datos...")

    # Asegurar que las columnas numéricas sean float
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

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

    print(f"Inconsistencias corregidas: {mask_inconsistent.sum()}")

    print(f"Dataset después de limpieza: {df_clean.shape}")
    return df_clean
