"""Pipeline de preparacion y entrenamiento de modelos."""

import logging
import os

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. PREPARACION DE DATOS
# =============================================================================


def prepare_features_target(
    df, target_column="Total Spent", exclude_columns=None
):
    """Preparar caracteristicas (X) y variable objetivo (y) para modelo."""
    if exclude_columns is None:
        exclude_columns = [
            "Transaction ID",
            "Customer ID",
            "Transaction Date",
            target_column,
        ]

    # Identificar columnas a excluir
    columns_to_exclude = [col for col in exclude_columns if col in df.columns]

    # Caracteristicas (X) - excluir columnas no deseadas y la variable objetivo
    feature_columns = [
        col
        for col in df.columns
        if col not in columns_to_exclude and col != target_column
    ]
    X = df[feature_columns]

    # Variable objetivo (y)
    y = df[target_column]

    logger.info(f"Caracteristicas: {X.shape[1]} columnas")
    logger.info(f"Variable objetivo: {y.name}")

    return X, y, feature_columns


def split_data(X, y, test_size=0.2, random_state=42):
    """Dividir datos en train/test (80/20)."""
    logger.info(
        f"\n📊 Dividiendo datos (train: {(1-test_size)*100:.0f}%, "
        f"test: {test_size*100:.0f}%)..."
    )

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train set: {X_train.shape}")
    logger.info(f"Test set: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def prepare_training_data(
    df_processed,
    target_column="Total Spent",
    exclude_columns=None,
    output_dir="data/processed",
    test_size=0.2,
):
    """Preparar datos para entrenamiento con division train/test.

    Args
    ----
        df_processed: DataFrame con datos preprocesados
        target_column: Nombre de la columna objetivo a predecir
        exclude_columns: Lista de columnas a excluir de las caracteristicas
        output_dir: Directorio donde guardar los archivos
        test_size: Proporcion de datos para test (default 0.2 = 20%)

    Returns
    -------
        dict con X_train, X_test, y_train, y_test, feature_columns

    """
    if exclude_columns is None:
        exclude_columns = [
            "Transaction ID",
            "Customer ID",
            "Transaction Date",
            target_column,
        ]

    logger.info("\n🎯 INICIANDO PREPARACION DE DATOS PARA ENTRENAMIENTO")
    logger.info("=" * 60)
    logger.info(f"Variable objetivo: {target_column}")
    logger.info(f"Columnas excluidas: {len(exclude_columns)}")

    # Paso 1: Preparar caracteristicas y variable objetivo
    logger.info("\n1️⃣ Separando caracteristicas y variable objetivo...")
    X, y, feature_columns = prepare_features_target(
        df_processed,
        target_column=target_column,
        exclude_columns=exclude_columns,
    )

    # Paso 2: Dividir en train/test
    logger.info("\n2️⃣ Dividiendo datos en train/test...")
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=42
    )

    # Paso 3: Guardar datasets
    logger.info("\n3️⃣ Guardando datasets...")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_columns": feature_columns,
    }


# =============================================================================
# 2. ENTRENAMIENTO DE MODELOS
# =============================================================================


def train_discount_applied_model(df_processed, test_size=0.2, max_iter=5000):
    """Entrenar modelo de clasificacion para predecir Discount Applied.

    Args
    ----
        df_processed: DataFrame con datos preprocesados
        test_size: Proporcion de datos para test (default 0.2)
        max_iter: Numero maximo de iteraciones para LogisticRegression

    Returns
    -------
        dict con model, metrics, y data splits

    """
    logger.info("\n" + "=" * 70)
    logger.info("🤖 ENTRENANDO MODELO: DISCOUNT APPLIED (CLASIFICACION)")
    logger.info("=" * 70)

    # Configurar parametros
    target_column = "Discount Applied"
    exclude_columns = [
        "Transaction ID",
        "Customer ID",
        "Transaction Date",
        "Discount Applied",
    ]

    # Preparar datos
    result = prepare_training_data(
        df_processed,
        target_column=target_column,
        exclude_columns=exclude_columns,
        output_dir="data/models/discount_applied",
        test_size=test_size,
    )

    # Entrenar modelo
    logger.info("\n4️⃣ Entrenando modelo de Regresion Logistica...")
    model = LogisticRegression(max_iter=max_iter, random_state=42)
    model.fit(result["X_train"], result["y_train"])
    logger.info("✅ Modelo entrenado exitosamente")

    # Evaluar modelo
    logger.info("\n5️⃣ Evaluando modelo...")
    y_pred_train = model.predict(result["X_train"])
    y_pred_test = model.predict(result["X_test"])

    train_accuracy = accuracy_score(result["y_train"], y_pred_train)
    test_accuracy = accuracy_score(result["y_test"], y_pred_test)

    logger.info(f"Exactitud en train: {train_accuracy:.4f}")
    logger.info(f"Exactitud en test: {test_accuracy:.4f}")

    # Guardar modelo
    logger.info("\n6️⃣ Guardando modelo...")
    model_dir = "data/models"
    os.makedirs(model_dir, exist_ok=True)

    # Guardar modelo de clasificacion
    model_path = os.path.join(model_dir, "discount_applied_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"✅ Modelo guardado en: {model_path}")

    # Guardar feature columns para usar en predicciones
    feature_columns_path = os.path.join(
        model_dir, "discount_applied_features.pkl"
    )
    joblib.dump(result["feature_columns"], feature_columns_path)
    logger.info(f"✅ Feature columns guardadas en: {feature_columns_path}")

    return {
        "model": model,
        "model_path": model_path,
        "metrics": {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        },
        "data": result,
    }


def train_total_spent_model(df_processed, test_size=0.2):
    """Entrenar modelo de regresion para predecir Total Spent.

    Args
    ----
        df_processed: DataFrame con datos preprocesados
        test_size: Proporcion de datos para test (default 0.2)

    Returns
    -------
        dict con model, metrics, y data splits

    """
    logger.info("\n" + "=" * 70)
    logger.info("🤖 ENTRENANDO MODELO: TOTAL SPENT (REGRESION)")
    logger.info("=" * 70)

    # Configurar parametros
    target_column = "Total Spent"
    exclude_columns = [
        "Transaction ID",
        "Customer ID",
        "Transaction Date",
        "Total Spent",
    ]

    # Preparar datos
    result = prepare_training_data(
        df_processed,
        target_column=target_column,
        exclude_columns=exclude_columns,
        output_dir="data/models/total_spent",
        test_size=test_size,
    )

    # Entrenar modelo
    logger.info("\n4️⃣ Entrenando modelo de Regresion Lineal...")
    model = LinearRegression()
    model.fit(result["X_train"], result["y_train"])
    logger.info("✅ Modelo entrenado exitosamente")

    # Evaluar modelo
    logger.info("\n5️⃣ Evaluando modelo...")
    y_pred_train = model.predict(result["X_train"])
    y_pred_test = model.predict(result["X_test"])

    train_mse = mean_squared_error(result["y_train"], y_pred_train)
    test_mse = mean_squared_error(result["y_test"], y_pred_test)
    train_r2 = r2_score(result["y_train"], y_pred_train)
    test_r2 = r2_score(result["y_test"], y_pred_test)

    logger.info(f"MSE en train: {train_mse:.4f}")
    logger.info(f"MSE en test: {test_mse:.4f}")
    logger.info(f"R2 en train: {train_r2:.4f}")
    logger.info(f"R2 en test: {test_r2:.4f}")

    # Guardar modelo
    logger.info("\n6️⃣ Guardando modelo...")
    model_dir = "data/models"
    os.makedirs(model_dir, exist_ok=True)

    # Guardar modelo de regresion
    model_path = os.path.join(model_dir, "total_spent_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"✅ Modelo guardado en: {model_path}")

    # Guardar feature columns para usar en predicciones
    feature_columns_path = os.path.join(model_dir, "total_spent_features.pkl")
    joblib.dump(result["feature_columns"], feature_columns_path)
    logger.info(f"✅ Feature columns guardadas en: {feature_columns_path}")

    return {
        "model": model,
        "model_path": model_path,
        "metrics": {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_r2": train_r2,
            "test_r2": test_r2,
        },
        "data": result,
    }


# =============================================================================
# 3. CARGA DE MODELOS
# =============================================================================


def load_discount_applied_model(model_dir="data/models"):
    """Cargar modelo de Discount Applied desde archivo.

    Args
    ----
        model_dir: Directorio donde estan los modelos

    Returns
    -------
        tuple (model, feature_columns)

    """
    model_path = os.path.join(model_dir, "discount_applied_model.pkl")
    features_path = os.path.join(model_dir, "discount_applied_features.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")

    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)

    logger.info(f"✅ Modelo cargado desde: {model_path}")
    logger.info(f"✅ Features cargadas: {len(feature_columns)} columnas")

    return model, feature_columns


def load_total_spent_model(model_dir="data/models"):
    """Cargar modelo de Total Spent desde archivo.

    Args
    ----
        model_dir: Directorio donde estan los modelos

    Returns
    -------
        tuple (model, feature_columns)

    """
    model_path = os.path.join(model_dir, "total_spent_model.pkl")
    features_path = os.path.join(model_dir, "total_spent_features.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")

    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)

    logger.info(f"✅ Modelo cargado desde: {model_path}")
    logger.info(f"✅ Features cargadas: {len(feature_columns)} columnas")

    return model, feature_columns


# =============================================================================
# 4. FUNCIONES PARA API
# =============================================================================


def predict_discount_applied(model, features):
    """Predecir si se aplicara descuento (para API).

    Args
    ----
        model: Modelo entrenado de LogisticRegression
        features: DataFrame o array con caracteristicas

    Returns
    -------
        dict con prediccion y probabilidad

    """
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return {
        "discount_applied": bool(prediction),
        "probability_false": float(probability[0]),
        "probability_true": float(probability[1]),
    }


def predict_total_spent(model, features):
    """Predecir el total gastado (para API).

    Args
    ----
        model: Modelo entrenado de LinearRegression
        features: DataFrame o array con caracteristicas

    Returns
    -------
        dict con prediccion

    """
    prediction = model.predict(features)[0]

    return {"total_spent": float(prediction)}


# =============================================================================
# 5. MAIN - EJEMPLO DE USO
# =============================================================================


if __name__ == "__main__":
    # Cargar datos procesados
    input_file = "data/processed/data_processed.parquet"

    logger.info(f"📥 Cargando datos procesados desde: {input_file}")
    df_processed = pd.read_parquet(input_file)

    # =========================================================================
    # OPCION 1: Entrenar modelo de Discount Applied
    # =========================================================================
    discount_result = train_discount_applied_model(
        df_processed, test_size=0.2, max_iter=5000
    )

    logger.info("\n" + "=" * 70)
    logger.info("✅ MODELO DISCOUNT APPLIED COMPLETADO")
    logger.info("=" * 70)
    logger.info(
        f"Train Accuracy: {discount_result['metrics']['train_accuracy']:.4f}"
    )
    logger.info(
        f"Test Accuracy: {discount_result['metrics']['test_accuracy']:.4f}"
    )

    # =========================================================================
    # OPCION 2: Entrenar modelo de Total Spent (comentado por defecto)
    # =========================================================================
    # total_spent_result = train_total_spent_model(df_processed, test_size=0.2)
    #
    # logger.info("\n" + "=" * 70)
    # logger.info("✅ MODELO TOTAL SPENT COMPLETADO")
    # logger.info("=" * 70)
    # logger.info(f"Train R2: {total_spent_result['metrics']['train_r2']:.4f}")
    # logger.info(f"Test R2: {total_spent_result['metrics']['test_r2']:.4f}")

    # =========================================================================
    # EJEMPLO DE PREDICCION (para API)
    # =========================================================================
    # # Tomar una muestra para probar prediccion
    # sample = discount_result['data']['X_test'].iloc[0:1]
    # prediction = predict_discount_applied(discount_result['model'], sample)
    # logger.info(f"\n📊 Ejemplo de prediccion: {prediction}")
