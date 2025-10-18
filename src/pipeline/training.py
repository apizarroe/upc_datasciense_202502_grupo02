"""Pipeline de preparacion y entrenamiento de modelos."""

import logging
import os
from typing import Optional, Tuple

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

# Configurar logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. PREPARACION DE DATOS
# =============================================================================


def prepare_features_target(
    df: pd.DataFrame,
    target_column: str = "Total Spent",
    exclude_columns: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.Series, list]:
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


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple:
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
    df_processed: pd.DataFrame,
    target_column: str = "Total Spent",
    exclude_columns: Optional[list] = None,
    output_dir: str = "data/processed",
    test_size: float = 0.2,
) -> dict:
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


def optimize_discount_model_hyperparameters(
    df_processed: pd.DataFrame,
    test_size: float = 0.2,
    cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 2,
) -> dict:
    """Optimizar hiperparametros del modelo usando GridSearchCV.

    Args
    ----
        df_processed: DataFrame con datos preprocesados
        test_size: Proporcion de datos para test (default 0.2)
        cv: Numero de folds para cross-validation (default 5)
        n_jobs: Numero de trabajos en paralelo (-1 = todos los cores)
        verbose: Nivel de verbosidad (0, 1, 2, 3)

    Returns
    -------
        dict con best_model, best_params, cv_results, y metrics

    """
    logger.info("\n" + "=" * 70)
    logger.info("🔍 OPTIMIZACION DE HIPERPARAMETROS - GRID SEARCH CV")
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

    # Definir el grid de hiperparametros a explorar
    # Grid enfocado en encontrar balance entre complejidad y generalizacion
    param_grid: dict = {
        "max_depth": [5, 10, 15, 20, None],  # Rango medio-alto
        "min_samples_split": [2, 5, 10, 15],  # Menos restrictivo
        "min_samples_leaf": [1, 2, 5, 8],  # Menos restrictivo
        "criterion": ["gini", "entropy"],
        "max_features": [None, "sqrt"],  # Simplificado
        "class_weight": [None, "balanced"],
        "ccp_alpha": [0.0, 0.001],  # Sin poda agresiva
    }

    logger.info("\n📋 Grid de hiperparametros:")
    for param, values in param_grid.items():
        logger.info(f"   - {param}: {values}")

    total_combinations = (
        len(param_grid["max_depth"])
        * len(param_grid["min_samples_split"])
        * len(param_grid["min_samples_leaf"])
        * len(param_grid["criterion"])
        * len(param_grid["max_features"])
        * len(param_grid["class_weight"])
        * len(param_grid["ccp_alpha"])
    )
    logger.info(f"\n⏱️  Total de combinaciones a probar: {total_combinations}")

    # Crear el modelo base
    base_model = DecisionTreeClassifier(random_state=42)

    # Configurar GridSearchCV
    logger.info(
        f"\n🔄 Ejecutando Grid Search con {cv}-fold cross-validation..."
    )
    logger.info("   (Esto puede tomar varios minutos...)\n")

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )

    # Entrenar con grid search
    grid_search.fit(result["X_train"], result["y_train"])

    # Obtener mejores parametros
    logger.info("\n" + "=" * 70)
    logger.info("✅ OPTIMIZACION COMPLETADA")
    logger.info("=" * 70)
    logger.info("\n🏆 Mejores hiperparametros encontrados:")
    for param, value in grid_search.best_params_.items():
        logger.info(f"   - {param}: {value}")

    # Evaluar el mejor modelo
    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(result["X_train"])
    y_pred_test = best_model.predict(result["X_test"])

    train_accuracy = accuracy_score(result["y_train"], y_pred_train)
    test_accuracy = accuracy_score(result["y_test"], y_pred_test)

    logger.info("\n📊 Metricas del mejor modelo:")
    logger.info(f"   - CV Score: {grid_search.best_score_:.4f}")
    logger.info(f"   - Train Accuracy: {train_accuracy:.4f}")
    logger.info(f"   - Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"   - Profundidad del arbol: {best_model.get_depth()}")
    logger.info(f"   - Numero de hojas: {best_model.get_n_leaves()}")

    # Reporte de clasificacion detallado
    logger.info("\n📈 Reporte de clasificacion (Test set):")
    logger.info("\n" + classification_report(result["y_test"], y_pred_test))

    # Guardar modelo optimizado
    logger.info("\n💾 Guardando modelo optimizado...")
    model_dir = "data/models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(
        model_dir, "discount_applied_model_optimized.pkl"
    )
    joblib.dump(best_model, model_path)
    logger.info(f"✅ Modelo optimizado guardado en: {model_path}")

    # Guardar mejores parametros
    params_path = os.path.join(model_dir, "best_hyperparameters.pkl")
    joblib.dump(grid_search.best_params_, params_path)
    logger.info(f"✅ Mejores parametros guardados en: {params_path}")

    # Guardar feature columns
    feature_columns_path = os.path.join(
        model_dir, "discount_applied_features.pkl"
    )
    joblib.dump(result["feature_columns"], feature_columns_path)

    return {
        "best_model": best_model,
        "model_path": model_path,
        "best_params": grid_search.best_params_,
        "cv_score": grid_search.best_score_,
        "cv_results": grid_search.cv_results_,
        "metrics": {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        },
        "data": result,
        "grid_search": grid_search,
    }


def train_discount_applied_model(
    df_processed: pd.DataFrame,
    test_size: float = 0.2,
    # Hiperparámetros del árbol de decisión
    # (valores balanceados por defecto)
    max_depth: Optional[int] = 15,  # Balance complejidad/generalizacion
    min_samples_split: int = 10,  # Menos restrictivo
    min_samples_leaf: int = 5,  # Menos restrictivo
    max_features: Optional[str] = None,  # Usar todas las features
    criterion: str = "gini",  # Gini generalmente funciona bien
    splitter: str = "best",  # Mejor split en cada nodo
    max_leaf_nodes: Optional[int] = None,  # Sin limite de hojas
    min_impurity_decrease: float = 0.0,  # Sin umbral minimo
    class_weight: Optional[str] = "balanced",  # Balancear clases
    ccp_alpha: float = 0.0,  # Sin poda inicial
    random_state: int = 42,
) -> dict:
    """Entrenar modelo de clasificacion para predecir Discount Applied.

    Args
    ----
        df_processed: DataFrame con datos preprocesados
        test_size: Proporcion de datos para test (default 0.2)
        max_depth: Profundidad maxima del arbol (None = sin limite)
        min_samples_split: Minimo de muestras requeridas para dividir
            nodo (default 2)
        min_samples_leaf: Minimo de muestras requeridas en nodo hoja
            (default 1)
        max_features: Numero maximo de features a considerar en cada
            split (None, 'sqrt', 'log2', int, float)
        criterion: Funcion para medir calidad del split
            ('gini' o 'entropy')
        splitter: Estrategia para dividir nodos ('best' o 'random')
        max_leaf_nodes: Numero maximo de nodos hoja (None = sin limite)
        min_impurity_decrease: Umbral minimo de reduccion de impureza
            para hacer split
        class_weight: Pesos de clases ('balanced', None, dict)
        ccp_alpha: Parametro de poda de complejidad (0.0 = sin poda)
        random_state: Semilla para reproducibilidad

    Returns
    -------
        dict con model, metrics, y data splits

    """
    logger.info("\n" + "=" * 70)
    logger.info("🌳 ENTRENANDO MODELO: DISCOUNT APPLIED (ARBOL DE DECISION)")
    logger.info("=" * 70)
    logger.info("\n📋 Hiperparametros configurados:")
    logger.info(f"   - max_depth: {max_depth}")
    logger.info(f"   - min_samples_split: {min_samples_split}")
    logger.info(f"   - min_samples_leaf: {min_samples_leaf}")
    logger.info(f"   - max_features: {max_features}")
    logger.info(f"   - criterion: {criterion}")
    logger.info(f"   - splitter: {splitter}")
    logger.info(f"   - max_leaf_nodes: {max_leaf_nodes}")
    logger.info(f"   - min_impurity_decrease: {min_impurity_decrease}")
    logger.info(f"   - class_weight: {class_weight}")
    logger.info(f"   - ccp_alpha: {ccp_alpha}")

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
    logger.info("\n4️⃣ Entrenando modelo de Arbol de Decision...")
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        criterion=criterion,
        splitter=splitter,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
        random_state=random_state,
    )
    model.fit(result["X_train"], result["y_train"])
    logger.info("✅ Modelo entrenado exitosamente")
    logger.info(f"   - Profundidad del arbol: {model.get_depth()}")
    logger.info(f"   - Numero de hojas: {model.get_n_leaves()}")

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


def train_total_spent_model(
    df_processed: pd.DataFrame, test_size: float = 0.2
) -> dict:
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


def load_discount_applied_model(
    model_dir: str = "data/models",
) -> Tuple[DecisionTreeClassifier, list]:
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


def load_total_spent_model(
    model_dir: str = "data/models",
) -> Tuple[LinearRegression, list]:
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


def predict_discount_applied(
    model: DecisionTreeClassifier, features: pd.DataFrame
) -> dict:
    """Predecir si se aplicara descuento (para API).

    Args
    ----
        model: Modelo entrenado (DecisionTreeClassifier)
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
        "confidence": float(max(probability)),  # Confianza de la prediccion
    }


def predict_total_spent(
    model: LinearRegression, features: pd.DataFrame
) -> dict:
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
    # ENTRENAR MODELO CON PARAMETROS CONFIGURABLES (para API)
    # =========================================================================
    discount_result = train_discount_applied_model(
        df_processed,
        test_size=0.2,
        # Parametros configurables - ajusta segun necesites:
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=None,
        criterion="gini",
        class_weight="balanced",
        ccp_alpha=0.0,
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
    # OPCION AVANZADA: Optimizar hiperparametros con GridSearchCV (comentado)
    # =========================================================================
    # Descomenta si quieres ejecutar busqueda automatica de mejores parametros:
    #
    # optimization_result = optimize_discount_model_hyperparameters(
    #     df_processed,
    #     test_size=0.2,
    #     cv=5,
    #     n_jobs=-1,
    #     verbose=2,
    # )
    #
    # logger.info("\n" + "=" * 70)
    # logger.info("✅ OPTIMIZACION COMPLETADA")
    # logger.info("=" * 70)
    # logger.info(f"CV Score: {optimization_result['cv_score']:.4f}")
    # train_acc = optimization_result['metrics']['train_accuracy']
    # logger.info(f"Train Accuracy: {train_acc:.4f}")
    # test_acc = optimization_result['metrics']['test_accuracy']
    # logger.info(f"Test Accuracy: {test_acc:.4f}")
    # logger.info(f"\n🏆 Mejores parametros encontrados:")
    # for param, value in optimization_result['best_params'].items():
    #     logger.info(f"   {param}: {value}")

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
