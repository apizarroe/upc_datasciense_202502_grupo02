"""Módulo para preprocesamiento de datos y preparación de modelos."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_for_regression(df, target_column="Total Spent"):
    """Preprocesamiento para modelo de regresión (predecir Total Spent)."""
    df_processed = df.copy()

    print("Preprocesamiento para predicción de Total Spent...")

    # 1. Codificación de variables categóricas
    categorical_columns = ["Category", "Item", "Payment Method", "Location"]
    label_encoders = {}

    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[f"{col}_encoded"] = le.fit_transform(
                df_processed[col].astype(str)
            )
            label_encoders[col] = le

    # 2. Extracción de características de fecha
    if "Transaction Date" in df_processed.columns:
        print("Extrayendo características de fecha...")

        df_processed["Transaction_Year"] = df_processed[
            "Transaction Date"
        ].dt.year
        df_processed["Transaction_Month"] = df_processed[
            "Transaction Date"
        ].dt.month
        df_processed["Transaction_Day"] = df_processed[
            "Transaction Date"
        ].dt.day
        df_processed["Transaction_DayOfWeek"] = df_processed[
            "Transaction Date"
        ].dt.dayofweek
        df_processed["Transaction_Quarter"] = df_processed[
            "Transaction Date"
        ].dt.quarter
        df_processed["Transaction_IsWeekend"] = (
            df_processed["Transaction_DayOfWeek"] >= 5
        ).astype(int)

    # 3. Creación de nuevas características relevantes para Total Spent
    print("Creando nuevas características...")

    # Interacción entre precio y cantidad
    df_processed["Expected_Total"] = (
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

    # Codificación de Transaction_Size
    le_size = LabelEncoder()
    df_processed["Transaction_Size_encoded"] = le_size.fit_transform(
        df_processed["Transaction_Size"]
    )
    label_encoders["Transaction_Size"] = le_size

    # 4. Codificación one-hot para variables categóricas importantes
    categorical_for_onehot = ["Category", "Location", "Payment Method"]

    for col in categorical_for_onehot:
        if col in df_processed.columns:
            dummies = pd.get_dummies(df_processed[col], prefix=col)
            df_processed = pd.concat([df_processed, dummies], axis=1)

    # 5. Codificación booleana
    df_processed["Discount_Applied"] = df_processed["Discount Applied"].astype(
        int
    )

    print(f"Dataset después de preprocesamiento: {df_processed.shape}")
    return df_processed, label_encoders


def prepare_features_target(
    df, target_column="Total Spent", exclude_columns=None
):
    """Preparar características (X) y variable objetivo (y) para modelo."""
    if exclude_columns is None:
        exclude_columns = [
            "Transaction ID",
            "Customer ID",
            "Transaction Date",
            target_column,
        ]

    # Identificar columnas a excluir
    columns_to_exclude = [col for col in exclude_columns if col in df.columns]

    # Características (X) - excluir columnas no deseadas y la variable objetivo
    feature_columns = [
        col
        for col in df.columns
        if col not in columns_to_exclude and col != target_column
    ]
    X = df[feature_columns]

    # Variable objetivo (y)
    y = df[target_column]

    print(f"Características: {X.shape[1]} columnas")
    print(f"Variable objetivo: {y.name}")

    return X, y, feature_columns


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """Dividir datos en train/test y escalar características numéricas."""
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    # Identificar columnas numéricas
    numeric_columns = X_train.select_dtypes(include=[np.number]).columns

    # Escalar características numéricas
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_columns] = scaler.fit_transform(
        X_train[numeric_columns]
    )
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
