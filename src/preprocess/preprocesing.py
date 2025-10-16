"""Módulo para preprocesamiento OPTIMIZADO de datos."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_for_regression(df, target_column="Total Spent"):
    """Preprocesamiento OPTIMIZADO para modelo de regresión."""
    df_processed = df.copy()

    print("Preprocesamiento OPTIMIZADO para predicción de Total Spent...")

    # 1. SOLO codificar categóricas esenciales
    essential_categoricals = ["Category", "Payment Method", "Location"]
    label_encoders = {}

    for col in essential_categoricals:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[f"{col}_encoded"] = le.fit_transform(
                df_processed[col].astype(str)
            )
            label_encoders[col] = le

    # 2. Extracción de características temporales ESENCIALES
    if "Transaction Date" in df_processed.columns:
        df_processed["Transaction_DayOfWeek"] = df_processed["Transaction Date"].dt.dayofweek
        df_processed["Transaction_IsWeekend"] = (df_processed["Transaction_DayOfWeek"] >= 5).astype(int)

    # 3. SOLO características derivadas CLAVE
    # Interacción precio-cantidad (MUY importante)
    df_processed["Price_Quantity_Interaction"] = (
        df_processed["Price Per Unit"] * df_processed["Quantity"]
    )

    # 4. Codificación booleana simple
    df_processed["Discount_Applied"] = df_processed["Discount Applied"].astype(int)

    print(f"Dataset después de preprocesamiento OPTIMIZADO: {df_processed.shape}")
    return df_processed, label_encoders

def prepare_features_optimized(df, target_column="Total Spent"):
    """Preparar características OPTIMIZADAS evitando data leakage."""
    print("Preparando características OPTIMIZADAS...")
    
    # DEFINIR MANUALMENTE las características que SÍ usaremos
    selected_features = [
        # Variables base
        'Price Per Unit', 'Quantity', 'Discount_Applied',
        
        # Interacciones
        'Price_Quantity_Interaction',
        
        # Características temporales
        'Transaction_DayOfWeek', 'Transaction_IsWeekend',
        
        # Categóricas codificadas
        'Category_encoded', 'Payment Method_encoded', 'Location_encoded',
        
        # Características de cliente OPTIMIZADAS
        'Customer_Transaction_Count', 'Customer_Avg_Spent', 
        'Customer_Discount_Frequency', 'Customer_Recency',
        
        # Características de producto OPTIMIZADAS  
        'Product_Transaction_Count', 'Product_Avg_Price',
        'Product_Discount_Rate', 'Product_Popularity_Score'
    ]
    
    # Filtrar solo las que existen en el dataset
    existing_features = [f for f in selected_features if f in df.columns]
    
    # Agregar el target
    if target_column in df.columns:
        existing_features.append(target_column)
    
    # Crear dataset optimizado
    df_optimized = df[existing_features]
    
    # Separar X e y
    X = df_optimized.drop(target_column, axis=1)
    y = df_optimized[target_column]
    
    print(f"✅ Características OPTIMIZADAS: {X.shape[1]} variables")
    print(f"✅ Registros: {X.shape[0]} transacciones")
    
    # Mostrar las características seleccionadas
    print("\n🔍 Características seleccionadas:")
    for i, col in enumerate(X.columns, 1):
        print(f"   {i:2d}. {col}")
    
    return X, y, X.columns.tolist()

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

    X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler