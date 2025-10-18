"""Modulo para realizar predicciones con modelos entrenados."""

import logging
import os
from datetime import datetime

import joblib
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. CARGAR MODELO Y ARTEFACTOS
# =============================================================================


def load_model_artifacts(model_dir="data/models"):
    """Cargar modelo y artefactos necesarios para prediccion.

    Args
    ----
        model_dir: Directorio donde estan los modelos

    Returns
    -------
        dict con model, feature_columns, y metadata

    """
    logger.info(f"Cargando artefactos del modelo desde: {model_dir}")

    # Cargar modelo
    model_path = os.path.join(model_dir, "discount_applied_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")

    model = joblib.load(model_path)
    logger.info("Modelo cargado exitosamente")

    # Cargar feature columns
    features_path = os.path.join(model_dir, "discount_applied_features.pkl")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features no encontradas en: {features_path}")

    feature_columns = joblib.load(features_path)
    logger.info(f"Features cargadas: {len(feature_columns)} columnas")

    return {
        "model": model,
        "feature_columns": feature_columns,
        "model_path": model_path,
    }


# =============================================================================
# 2. PREPROCESAR DATOS PARA PREDICCION
# =============================================================================


def preprocess_for_prediction(data, feature_columns):
    """Preparar datos preprocesados para prediccion.

    IMPORTANTE: Esta funcion asume que los datos YA ESTAN preprocesados
    (escalados, codificados, etc.) como en data_for_prediction.parquet

    Args
    ----
        data: dict o DataFrame con datos preprocesados
        feature_columns: Lista de columnas que espera el modelo

    Returns
    -------
        DataFrame con features en el orden correcto

    """
    logger.info("Validando datos para prediccion...")

    # Convertir a DataFrame si es dict
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    logger.info(f"Datos de entrada: {df.shape}")

    # Validar que todas las feature columns existan
    missing_features = set(feature_columns) - set(df.columns)

    if missing_features:
        logger.warning(
            f"Features faltantes: {missing_features}. Rellenando con 0."
        )
        for col in missing_features:
            df[col] = 0

    # Seleccionar solo las columnas que espera el modelo, en el orden correcto
    df_processed = df[feature_columns]

    logger.info(f"Datos listos para prediccion: {df_processed.shape}")

    return df_processed


# =============================================================================
# 3. FUNCIONES DE PREDICCION
# =============================================================================


def predict_discount_applied(data, model_dir="data/models"):
    """Predecir si se aplicara descuento (para uso en API).

    Args
    ----
        data: dict o DataFrame con datos de la transaccion
        model_dir: Directorio donde estan los modelos

    Returns
    -------
        dict con prediccion, probabilidades y metadata

    Ejemplo de data (dict)
    ----------------------
        {
            "Price Per Unit": 50.0,
            "Quantity": 3,
            "Category_0": 1,
            "Category_1": 0,
            # ... todas las features necesarias
        }

    """
    # Cargar artefactos
    artifacts = load_model_artifacts(model_dir)
    model = artifacts["model"]
    feature_columns = artifacts["feature_columns"]

    # Preprocesar datos
    df_processed = preprocess_for_prediction(data, feature_columns)

    # Predecir
    logger.info("Realizando prediccion...")
    prediction = model.predict(df_processed)[0]
    probabilities = model.predict_proba(df_processed)[0]

    # Construir respuesta
    result = {
        "discount_applied": bool(prediction),
        "prediction": int(prediction),
        "probability_no_discount": float(probabilities[0]),
        "probability_discount": float(probabilities[1]),
        "confidence": float(max(probabilities)),
        "model_info": {
            "model_type": type(model).__name__,
            "n_features": len(feature_columns),
            "model_depth": model.get_depth() if hasattr(model, "get_depth") else None,
            "n_leaves": model.get_n_leaves() if hasattr(model, "get_n_leaves") else None,
        },
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(
        f"Prediccion: {result['discount_applied']} "
        f"(confianza: {result['confidence']:.2%})"
    )

    return result


def predict_batch(data_list, model_dir="data/models", save_results=True):
    """Predecir multiples transacciones en batch.

    Args
    ----
        data_list: Lista de dicts con datos de transacciones
        model_dir: Directorio donde estan los modelos
        save_results: Si True, guarda resultados en data/processed/data_predicted.parquet

    Returns
    -------
        list de dicts con predicciones

    """
    logger.info(f"Prediccion en batch: {len(data_list)} transacciones")

    # Cargar artefactos una sola vez
    artifacts = load_model_artifacts(model_dir)
    model = artifacts["model"]
    feature_columns = artifacts["feature_columns"]

    # Convertir a DataFrame
    df = pd.DataFrame(data_list)

    # Preprocesar
    df_processed = preprocess_for_prediction(df, feature_columns)

    # Predecir
    predictions = model.predict(df_processed)
    probabilities = model.predict_proba(df_processed)

    # Construir respuestas
    results = []
    for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
        result = {
            "index": i,
            "discount_applied": bool(pred),
            "prediction": int(pred),
            "probability_no_discount": float(probs[0]),
            "probability_discount": float(probs[1]),
            "confidence": float(max(probs)),
        }
        results.append(result)

    logger.info(f"Predicciones completadas: {len(results)}")

    # Guardar resultados si se solicita
    if save_results:
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "data_predicted.parquet")

        df_results = pd.DataFrame(results)
        df_results.to_parquet(output_path, index=False)
        logger.info(f"Resultados guardados en: {output_path}")

    return results


# =============================================================================
# 4. FUNCIONES PARA API
# =============================================================================


def predict_from_parquet(
    parquet_path="data/processed/data_processed.parquet",
    model_dir="data/models",
    row_index=None,
    save_results=True,
):
    """Predecir desde archivo parquet (simula llamada de API).

    Args
    ----
        parquet_path: Ruta al archivo parquet con datos preprocesados
        model_dir: Directorio donde estan los modelos
        row_index: Indice de fila especifica (None = todas las filas)
        save_results: Si True, guarda resultados batch en data_predicted.parquet

    Returns
    -------
        dict o list con predicciones

    """
    logger.info(f"Cargando datos desde: {parquet_path}")

    # Cargar datos
    df = pd.read_parquet(parquet_path)
    logger.info(f"Datos cargados: {df.shape}")

    # Si se especifica un indice, tomar solo esa fila
    if row_index is not None:
        if row_index >= len(df):
            raise ValueError(
                f"Indice {row_index} fuera de rango. Dataset tiene {len(df)} filas."
            )
        data = df.iloc[row_index : row_index + 1]
        logger.info(f"Prediciendo fila #{row_index}")
        return predict_discount_applied(data, model_dir=model_dir)
    else:
        # Predecir todas las filas
        logger.info(f"Prediciendo {len(df)} filas en batch")
        return predict_batch(
            df.to_dict("records"), model_dir=model_dir, save_results=save_results
        )


def predict_from_dict(data_dict, model_dir="data/models"):
    """Predecir desde diccionario (simula request de API REST).

    Args
    ----
        data_dict: Diccionario con features preprocesadas
        model_dir: Directorio donde estan los modelos

    Returns
    -------
        dict con prediccion

    Ejemplo de data_dict
    --------------------
        {
            "Age": 0.5234,
            "Quantity": 1.234,
            "Price Per Unit": -0.456,
            "Category_0": 1,
            "Category_1": 0,
            "Gender_Male": 1,
            # ... todas las features preprocesadas
        }

    """
    logger.info("Prediccion desde diccionario (API request)")
    return predict_discount_applied(data_dict, model_dir=model_dir)


# =============================================================================
# 5. EJEMPLO DE USO
# =============================================================================


if __name__ == "__main__":
    logger.info("\n" + "=" * 70)
    logger.info("EJEMPLO DE PREDICCION - SIMULACION DE API")
    logger.info("=" * 70)

    try:
        # Ejemplo 1: Prediccion individual (simula endpoint /predict/single)
        logger.info("\n1️⃣ Prediccion individual desde parquet...")
        parquet_path="data/processed/data_processed.parquet"

        df_processed = pd.read_parquet(parquet_path)
        df_processed = df_processed.drop(columns=["Discount Applied"])
        print(df_processed.info())


        result = predict_from_parquet(
            parquet_path=parquet_path,
            model_dir="data/models",
            row_index=0,  # Primera fila
        )

        logger.info("\n📊 Respuesta de API (JSON):")
        logger.info(f"   discount_applied: {result['discount_applied']}")
        logger.info(f"   probability_discount: {result['probability_discount']:.4f}")
        logger.info(f"   probability_no_discount: {result['probability_no_discount']:.4f}")
        logger.info(f"   confidence: {result['confidence']:.4f}")
        logger.info(f"   timestamp: {result['timestamp']}")
        logger.info(f"   model_type: {result['model_info']['model_type']}")

        # Ejemplo 2: Prediccion batch (simula endpoint /predict/batch)
        logger.info("\n2️⃣ Prediccion batch (primeras 10 filas)...")
        df = pd.read_parquet("data/processed/data_processed.parquet")

        if len(df) >= 10:
            batch_data = df.iloc[0:100].to_dict("records")
            results = predict_batch(batch_data, model_dir="data/models")

            logger.info("\n📊 Resultados batch (JSON):")
            discount_count = sum(1 for r in results if r["discount_applied"])
            logger.info(f"   Total predicciones: {len(results)}")
            logger.info(f"   Con descuento: {discount_count}")
            logger.info(f"   Sin descuento: {len(results) - discount_count}")

            logger.info("\n   Primeras 5 predicciones:")
            for r in results[:5]:
                logger.info(
                    f"      #{r['index']}: {r['discount_applied']} "
                    f"(prob: {r['probability_discount']:.2%})"
                )

        # Ejemplo 3: Prediccion desde diccionario (simula request JSON)
        logger.info("\n3️⃣ Prediccion desde diccionario (JSON request)...")

        # Tomar primera fila como ejemplo
        sample_dict = df.iloc[0].to_dict()
        result = predict_from_dict(sample_dict, model_dir="data/models")

        logger.info("\n📊 Request JSON simulado:")
        logger.info(f"   Numero de features: {len(sample_dict)}")
        logger.info("\n📊 Response JSON:")
        logger.info(f"   discount_applied: {result['discount_applied']}")
        logger.info(f"   confidence: {result['confidence']:.2%}")

        # Resumen
        logger.info("\n" + "=" * 70)
        logger.info("✅ EJEMPLOS COMPLETADOS")
        logger.info("=" * 70)
        logger.info("\n💡 Uso en API:")
        logger.info("   from src.pipeline.predict import predict_from_dict")
        logger.info("   result = predict_from_dict(request.json())")
        logger.info("   return JSONResponse(result)")

    except FileNotFoundError as e:
        logger.error(f"❌ Error: {e}")
        logger.info("\n💡 Ejecuta primero el pipeline de entrenamiento:")
        logger.info("   python src/pipeline/training.py")

    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
