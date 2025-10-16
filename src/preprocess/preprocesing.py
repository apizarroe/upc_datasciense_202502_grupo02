"""Módulo para preprocesamiento OPTIMIZADO de datos."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_for_regression(df, target_column="Total Spent"):
    """Preprocesamiento OPTIMIZADO para modelo de regresión."""
    df_processed = df.copy()

    print("Preprocesamiento OPTIMIZADO para predicción de Total Spent...")

    # 1. Manejar Discount Applied primero (convertir a numérico)
    if "Discount Applied" in df_processed.columns:
        # Convertir a booleano y luego a int (0 o 1)
        df_processed["Discount Applied"] = df_processed[
            "Discount Applied"
        ].fillna(False)
        df_processed["Discount Applied"] = (
            df_processed["Discount Applied"].astype(bool).astype(int)
        )

    # 2. SOLO codificar categóricas esenciales
    essential_categoricals = ["Category", "Payment Method", "Location"]
    label_encoders = {}

    for col in essential_categoricals:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[f"{col}"] = le.fit_transform(
                df_processed[col].astype(str)
            )
            label_encoders[col] = le

    # 2. Extracción de características temporales ESENCIALES
    if "Transaction Date" in df_processed.columns:
        print("Extrayendo características de fecha...")

        df_processed["Transaction_Year"] = df_processed[
            "Transaction Date"
        ].dt.year
        df_processed["Transaction_Month"] = df_processed[
            "Transaction Date"
        ].dt.month
        df_processed["Transaction_Quarter"] = df_processed[
            "Transaction Date"
        ].dt.quarter

    # 3. SOLO características derivadas CLAVE
    # Interacción precio-cantidad (MUY importante)
    df_processed["Price_Quantity_Interaction"] = (
        df_processed["Price Per Unit"] * df_processed["Quantity"]
    )

    # Ratio precio/cantidad
    df_processed["Price_Quantity_Ratio"] = df_processed["Price Per Unit"] / (
        df_processed["Quantity"] + 1
    )

    # Segmentación de transacciones por monto
    df_processed["Transaction_Size"] = pd.cut(
        df_processed["Total Spent"],
        bins=[0, 100, 500, 1000, float("inf")],
        labels=["Small", "Medium", "Large", "Very Large"],
    )

    # Codificación de Transaction_Size (convertir a numérico)
    if "Transaction_Size" in df_processed.columns:
        le_size = LabelEncoder()
        df_processed["Transaction_Size"] = le_size.fit_transform(
            df_processed["Transaction_Size"].astype(str)
        )
        label_encoders["Transaction_Size"] = le_size

    # 4. Codificación one-hot para variables categóricas importantes
    categorical_for_onehot = ["Category", "Location", "Payment Method"]

    for col in categorical_for_onehot:
        if col in df_processed.columns:
            dummies = pd.get_dummies(df_processed[col], prefix=col)
            df_processed = pd.concat([df_processed, dummies], axis=1)

    df_processed = df_processed.drop(
        columns=[
            "Transaction ID",
            "Category",
            "Item",
            "Payment Method",
            "Location",
        ]
    )

    # Escalado de los valores (EXCLUIR columnas one-hot y categoricas)
    # Identificar columnas one-hot (empiezan con prefijos de categorias)
    onehot_prefixes = ["Category_", "Location_", "Payment Method_"]
    onehot_cols = [
        col
        for col in df_processed.columns
        if any(col.startswith(prefix) for prefix in onehot_prefixes)
    ]

    # Columnas categoricas codificadas que NO deben escalarse
    categorical_encoded = ["Transaction_Size", "Discount Applied"]

    # Seleccionar solo columnas numericas continuas (NO one-hot, NO categoricas)
    numeric_cols = df_processed.select_dtypes(
        include=["int64", "float64", "number"]
    ).columns
    numeric_cols_to_scale = [
        col
        for col in numeric_cols
        if col not in onehot_cols and col not in categorical_encoded
    ]

    # Escalar solo las columnas numericas continuas
    scaler = StandardScaler()
    if len(numeric_cols_to_scale) > 0:
        df_processed[numeric_cols_to_scale] = scaler.fit_transform(
            df_processed[numeric_cols_to_scale]
        )

    print(f"Columnas numericas escaladas: {len(numeric_cols_to_scale)}")
    print(f"Columnas one-hot (NO escaladas): {len(onehot_cols)}")
    print(
        f"Columnas categoricas codificadas (NO escaladas): "
        f"{len(categorical_encoded)}"
    )

    # pd.set_option("display.max_columns", None)
    # print(df_processed)
    # pd.reset_option("display.max_columns")
    for col, le in label_encoders.items():
        if col == "Discount Applied":
            print(f"Columna: {col}")
            print(f"Clases: {list(le.classes_)}\n")

    print(
        f"Dataset después de preprocesamiento OPTIMIZADO: {df_processed.shape}"
    )
    return df_processed, label_encoders
