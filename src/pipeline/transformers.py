"""Pipeline completo de procesamiento de datos para ventas retail."""

import logging
import os
import sys

import pandas as pd

# Agregar src al path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from preprocess.agregation import (  # noqa: E402
    create_customer_features,
    create_product_features,
    enrich_transaction_data,
)
from preprocess.cleaning import clean_data  # noqa: E402
from preprocess.preprocesing import preprocess_for_classification  # noqa: E402

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(input_file, output_dir="data/processed"):
    """Pipeline completo de procesamiento de datos."""
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
        df_enriched = enrich_transaction_data(
            df_clean, customer_features, product_features
        )

        # 4. Preprocesamiento
        logger.info("\n⚙️ Realizando preprocesamiento...")
        df_processed, label_encoders = preprocess_for_classification(df_enriched)

        # Guardar datos procesados para entrenamiento
        processed_path = os.path.join(output_dir, "data_processed.parquet")
        df_processed.to_parquet(processed_path, index=False)
        logger.info("✅ data_processed.parquet")

        # 5. Resumen final
        logger.info("\n" + "=" * 50)
        logger.info("✅ PIPELINE DE PREPROCESAMIENTO COMPLETADO!")
        logger.info("=" * 50)
        logger.info(f"📁 Archivos guardados en: {output_dir}")
        logger.info(" - data_cleaned.parquet")
        logger.info(" - data_processed.parquet")

        return {
            "df_clean": df_clean,
            "df_processed": df_processed,
            "label_encoders": label_encoders,
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
