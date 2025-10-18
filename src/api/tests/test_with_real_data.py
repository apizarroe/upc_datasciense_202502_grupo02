"""Script para probar la API usando datos reales del dataset procesado."""

import os
from typing import Any, Optional, Tuple, Union

import pandas as pd
import requests

# URL de la API
API_URL = "http://localhost:8000"


def load_sample_data(
    n_samples: int = 5,
) -> Tuple[Optional[pd.DataFrame], Any]:
    """Cargar datos de muestra del dataset procesado."""
    try:
        print("📥 Cargando datos procesados...")

        # Ruta relativa desde src/api/tests/ a data/processed/
        data_path = os.path.join(
            os.path.dirname(__file__),
            "../../../data/processed/data_processed.parquet",
        )
        df = pd.read_parquet(data_path)

        # Eliminar columna target si existe
        if "Discount Applied" in df.columns:
            y_true = df["Discount Applied"].values[:n_samples]
            df = df.drop(columns=["Discount Applied"])
        else:
            y_true = None

        # Tomar primeras n_samples filas
        sample_df = df.head(n_samples)

        print(f"✅ Cargados {len(sample_df)} registros")
        print(f"   Columnas: {len(sample_df.columns)}")

        return sample_df, y_true

    except FileNotFoundError:
        print(
            "❌ Error: No se encontró "
            "data/processed/data_processed.parquet"
        )
        print("   Ejecuta primero el preprocesamiento de datos")
        return None, None
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return None, None


def test_with_real_data() -> None:  # noqa: C901
    """Probar la API con datos reales."""
    print("\n" + "=" * 70)
    print("🧪 TEST API CON DATOS REALES")
    print("=" * 70)

    # Verificar que la API esté disponible
    try:
        print("\n1️⃣  Verificando API...")
        response = requests.get(f"{API_URL}/health")
        if response.status_code != 200:
            print("❌ API no está disponible")
            return
        print("✅ API disponible")

        health_data = response.json()
        if not health_data["model_loaded"]:
            print("❌ Modelo no está cargado")
            print("   Ejecuta: python src/pipeline/training.py")
            return
        print(f"✅ Modelo cargado: {health_data['model_info']['model_type']}")

    except requests.exceptions.ConnectionError:
        print("❌ No se pudo conectar a la API")
        print("   Ejecuta: python run_api.py")
        return

    # Cargar datos de muestra
    print("\n2️⃣  Cargando datos de muestra...")
    sample_df, y_true = load_sample_data(n_samples=5)

    if sample_df is None:
        return

    # Test 1: Predicción individual
    print("\n" + "=" * 70)
    print("3️⃣  TEST: Predicción Individual (Primera fila)")
    print("=" * 70)

    first_row = sample_df.iloc[0].to_dict()

    response = requests.post(
        f"{API_URL}/predict", json={"features": first_row}
    )

    if response.status_code == 200:
        result = response.json()
        print("\n✅ Predicción exitosa:")
        print(f"   Descuento Aplicado: {result['discount_applied']}")
        print(f"   Confianza: {result['confidence']:.2%}")
        print(f"   Prob. Descuento: {result['probability_discount']:.4f}")
        print(
            f"   Prob. No Descuento: {result['probability_no_discount']:.4f}"
        )

        if y_true is not None:
            actual = bool(y_true[0])
            predicted = result["discount_applied"]
            match = "✅ CORRECTO" if actual == predicted else "❌ INCORRECTO"
            print(f"\n   Valor Real: {actual}")
            print(f"   Predicción: {predicted}")
            print(f"   {match}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())

    # Test 2: Predicción batch
    print("\n" + "=" * 70)
    print("4️⃣  TEST: Predicción Batch (5 filas)")
    print("=" * 70)

    batch_data = sample_df.to_dict("records")

    response = requests.post(
        f"{API_URL}/predict/batch", json={"data_list": batch_data}
    )

    if response.status_code == 200:
        result = response.json()
        print("\n✅ Predicción batch exitosa:")
        print(f"   Total: {result['total']}")
        print(f"   Con descuento: {result['discount_count']}")
        print(f"   Sin descuento: {result['no_discount_count']}")

        print("\n   Detalle de predicciones:")
        header = (
            f"   {'#':<4} {'Predicción':<12} {'Confianza':<12} "
            f"{'Real':<12} {'Match':<8}"
        )
        print(header)
        print(f"   {'-'*4} {'-'*12} {'-'*12} {'-'*12} {'-'*8}")

        correct_predictions = 0
        for pred in result["predictions"]:
            idx = pred["index"]
            predicted = pred["discount_applied"]
            confidence = pred["confidence"]

            if y_true is not None:
                actual = bool(y_true[idx])
                match = "✅" if actual == predicted else "❌"
                if actual == predicted:
                    correct_predictions += 1
            else:
                actual = "N/A"
                match = "-"

            print(
                f"   {idx:<4} {str(predicted):<12} {confidence:<12.2%} "
                f"{str(actual):<12} {match:<8}"
            )

        if y_true is not None:
            accuracy = correct_predictions / len(result["predictions"]) * 100
            print(
                f"\n   Accuracy: {accuracy:.1f}% "
                f"({correct_predictions}/{len(result['predictions'])})"
            )

    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())

    # Test 3: Información del modelo
    print("\n" + "=" * 70)
    print("5️⃣  TEST: Información del Modelo")
    print("=" * 70)

    response = requests.get(f"{API_URL}/model/info")

    if response.status_code == 200:
        info = response.json()
        print("\n✅ Información del modelo:")
        print(f"   Tipo: {info['model_type']}")
        print(f"   Features: {info['n_features']}")
        print(f"   Profundidad: {info['model_depth']}")
        print(f"   Hojas: {info['n_leaves']}")
        print("\n   Primeras 10 features esperadas:")
        for i, feature in enumerate(info["feature_columns"], 1):
            print(f"      {i}. {feature}")
    else:
        print(f"❌ Error: {response.status_code}")

    print("\n" + "=" * 70)
    print("✅ TESTS COMPLETADOS")
    print("=" * 70)


if __name__ == "__main__":
    test_with_real_data()
