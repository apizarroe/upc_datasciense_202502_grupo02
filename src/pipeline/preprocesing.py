import pandas as pd
import logging
import sys
import os

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Importar los módulos existentes
from preprocess.cleaning import clean_data
from preprocess.preprocesing import preprocess_for_regression, prepare_features_target
from preprocess.agregation import create_customer_features, create_product_features, enrich_transaction_data

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_pipeline(input_file, output_dir="data/processed"):
    """
    Pipeline completo de procesamiento de datos
    """
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("🚀 Iniciando pipeline de procesamiento de datos...")
    logger.info(f"Archivo de entrada: {input_file}")
    logger.info(f"Directorio de salida: {output_dir}")
    
    try:
        # 1. Cargar datos
        logger.info("\n📥 Cargando datos...")
        df = pd.read_csv(input_file)
        logger.info(f"Datos cargados: {df.shape}")
        
        # 2. Limpieza de datos
        logger.info("\n🧹 Realizando limpieza de datos...")
        df_clean = clean_data(df)
        
        # Guardar datos limpios
        clean_path = os.path.join(output_dir, "data_cleaned.parquet")
        df_clean.to_parquet(clean_path, index=False)
        logger.info("✅ data_cleaned.parquet")
        
        # 3. Crear características agregadas
        logger.info("\n📊 Creando características agregadas...")
        customer_features = create_customer_features(df_clean)
        product_features = create_product_features(df_clean)
        df_enriched = enrich_transaction_data(df_clean, customer_features, product_features)
        
        # 4. Preprocesamiento
        logger.info("\n⚙️ Realizando preprocesamiento...")
        df_processed, label_encoders = preprocess_for_regression(df_enriched)
        
        # Guardar datos procesados
        processed_path = os.path.join(output_dir, "data_processed.parquet")
        df_processed.to_parquet(processed_path, index=False)
        logger.info("✅ data_processed.parquet")
        
        # 5. Preparar datos para entrenamiento
        logger.info("\n🎯 Preparando datos para entrenamiento...")
        X, y, feature_columns = prepare_features_target(
            df_processed, 
            target_column='Total Spent',
            exclude_columns=['Transaction ID', 'Customer ID', 'Transaction Date', 'Total Spent']
        )
        
        # Crear dataset de entrenamiento (X + y)
        training_data = X.copy()
        training_data['Total_Spent'] = y
        
        # Guardar datos de entrenamiento
        training_path = os.path.join(output_dir, "data_training.parquet")
        training_data.to_parquet(training_path, index=False)
        logger.info("✅ data_training.parquet")
        
        # 6. Resumen final
        logger.info("\n" + "="*50)
        logger.info("✅ PIPELINE COMPLETADO EXITOSAMENTE!")
        logger.info("="*50)
        logger.info(f"📁 Archivos guardados en: {output_dir}")
        logger.info(" - data_cleaned.parquet")
        logger.info(" - data_processed.parquet") 
        logger.info(" - data_training.parquet")
        
        return {
            'df_clean': df_clean,
            'df_processed': df_processed,
            'training_data': training_data,
            'feature_columns': feature_columns,
            'label_encoders': label_encoders
        }
        
    except Exception as e:
        logger.error(f"❌ Error en el pipeline: {str(e)}")
        raise

if __name__ == "__main__":
     # Configurar rutas
    input_file = "data/raw/retail_store_sales.csv"  # Ajusta esta ruta
    output_dir = "data/processed"
    
    # Ejecutar pipeline
    run_pipeline(input_file, output_dir)