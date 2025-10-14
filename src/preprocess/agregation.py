import pandas as pd
import numpy as np

def create_customer_features(df):
    """
    Crear características agregadas por cliente para enriquecer el dataset
    """
    print("Creando características de cliente...")
    
    # Agregar por cliente
    customer_features = df.groupby('Customer ID').agg({
        'Transaction ID': 'count',
        'Total Spent': ['sum', 'mean', 'std', 'max'],
        'Quantity': ['sum', 'mean'],
        'Price Per Unit': 'mean',
        'Discount Applied': 'mean',  # Frecuencia de descuentos
        'Transaction Date': ['min', 'max'],
        'Category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',  # Categoría favorita
        'Location': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'   # Ubicación más frecuente
    }).round(2)
    
    # Aplanar columnas multi-index
    customer_features.columns = [
        'Customer_Transaction_Count',
        'Customer_Total_Spent_Sum', 'Customer_Avg_Spent', 'Customer_Std_Spent', 'Customer_Max_Spent',
        'Customer_Total_Quantity', 'Customer_Avg_Quantity',
        'Customer_Avg_Price_Per_Unit',
        'Customer_Discount_Frequency',
        'Customer_First_Transaction', 'Customer_Last_Transaction',
        'Customer_Favorite_Category', 'Customer_Preferred_Location'
    ]
    
    # Calcular métricas adicionales
    customer_features['Customer_Spent_Per_Transaction'] = (
        customer_features['Customer_Total_Spent_Sum'] / customer_features['Customer_Transaction_Count']
    )
    
    customer_features['Customer_Recency'] = (
        pd.to_datetime('today') - pd.to_datetime(customer_features['Customer_Last_Transaction'])
    ).dt.days
    
    # Segmentar clientes por valor
    customer_features['Customer_Value_Segment'] = pd.cut(
        customer_features['Customer_Total_Spent_Sum'],
        bins=[0, 1000, 5000, float('inf')],
        labels=['Low_Value', 'Medium_Value', 'High_Value']
    )
    
    return customer_features.reset_index()

def create_product_features(df):
    """
    Crear características agregadas por producto/categoría
    """
    print("Creando características de producto...")
    
    product_features = df.groupby(['Category', 'Item']).agg({
        'Transaction ID': 'count',
        'Total Spent': ['sum', 'mean', 'std'],
        'Quantity': ['sum', 'mean'],
        'Price Per Unit': ['mean', 'std'],
        'Discount Applied': 'mean'
    }).round(2)
    
    # Aplanar columnas
    product_features.columns = [
        'Product_Transaction_Count',
        'Product_Total_Revenue', 'Product_Avg_Revenue', 'Product_Std_Revenue',
        'Product_Total_Quantity', 'Product_Avg_Quantity',
        'Product_Avg_Price', 'Product_Std_Price',
        'Product_Discount_Rate'
    ]
    
    # Métricas adicionales
    product_features['Product_Popularity_Score'] = (
        product_features['Product_Transaction_Count'] / product_features['Product_Transaction_Count'].sum()
    )
    
    product_features['Product_Revenue_Per_Transaction'] = (
        product_features['Product_Total_Revenue'] / product_features['Product_Transaction_Count']
    )
    
    return product_features.reset_index()

def enrich_transaction_data(df, customer_features, product_features):
    """
    Enriquecer datos de transacción con características agregadas
    """
    print("Enriqueciendo datos de transacción...")
    
    df_enriched = df.copy()
    
    # Unir características de cliente
    df_enriched = df_enriched.merge(
        customer_features, 
        on='Customer ID', 
        how='left',
        suffixes=('', '_customer')
    )
    
    # Unir características de producto
    df_enriched = df_enriched.merge(
        product_features,
        on=['Category', 'Item'],
        how='left',
        suffixes=('', '_product')
    )
    
    print(f"Dataset enriquecido: {df_enriched.shape}")
    return df_enriched