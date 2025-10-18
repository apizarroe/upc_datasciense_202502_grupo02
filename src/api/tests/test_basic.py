"""Tests basicos para la API de prediccion de descuentos."""

import json

import requests

# URL de la API
API_URL = "http://localhost:8000"


def test_health():
    """Test del endpoint de health check."""
    print("\n" + "=" * 70)
    print("1️⃣  TEST: Health Check")
    print("=" * 70)

    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_model_info():
    """Test del endpoint de informacion del modelo."""
    print("\n" + "=" * 70)
    print("2️⃣  TEST: Model Info")
    print("=" * 70)

    response = requests.get(f"{API_URL}/model/info")
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"Model Type: {data['model_type']}")
        print(f"Number of Features: {data['n_features']}")
        print(f"Model Depth: {data['model_depth']}")
        print(f"Number of Leaves: {data['n_leaves']}")
        print(f"\nFirst 10 Features:")
        for i, feature in enumerate(data["feature_columns"], 1):
            print(f"  {i}. {feature}")
    else:
        print(f"Error: {response.json()}")

    return response.status_code == 200


def test_single_prediction():
    """Test del endpoint de prediccion individual."""
    print("\n" + "=" * 70)
    print("3️⃣  TEST: Single Prediction")
    print("=" * 70)

    # Ejemplo de features (ajusta según tu modelo)
    sample_features = {
        "Age": 0.5234,
        "Quantity": 1.234,
        "Price Per Unit": -0.456,
        "Category_0": 1,
        "Category_1": 0,
        "Gender_Male": 1,
    }

    print(f"\nRequest Features:")
    print(json.dumps(sample_features, indent=2))

    response = requests.post(
        f"{API_URL}/predict", json={"features": sample_features}
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nPrediction Results:")
        print(f"  Discount Applied: {result['discount_applied']}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Probability No Discount: {result['probability_no_discount']:.4f}")
        print(f"  Probability Discount: {result['probability_discount']:.4f}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Timestamp: {result['timestamp']}")
    else:
        print(f"Error: {response.json()}")

    return response.status_code == 200


def test_batch_prediction():
    """Test del endpoint de prediccion batch."""
    print("\n" + "=" * 70)
    print("4️⃣  TEST: Batch Prediction")
    print("=" * 70)

    # Ejemplo de multiples transacciones
    batch_data = [
        {
            "Age": 0.5234,
            "Quantity": 1.234,
            "Price Per Unit": -0.456,
            "Category_0": 1,
        },
        {
            "Age": -0.234,
            "Quantity": 2.567,
            "Price Per Unit": 0.123,
            "Category_0": 0,
        },
        {
            "Age": 0.789,
            "Quantity": 0.456,
            "Price Per Unit": 1.234,
            "Category_0": 1,
        },
    ]

    print(f"\nRequest: {len(batch_data)} items")

    response = requests.post(
        f"{API_URL}/predict/batch", json={"data_list": batch_data}
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print(f"\nBatch Prediction Results:")
        print(f"  Total: {result['total']}")
        print(f"  Discount Count: {result['discount_count']}")
        print(f"  No Discount Count: {result['no_discount_count']}")

        print(f"\n  Individual Predictions:")
        for pred in result["predictions"]:
            print(
                f"    #{pred['index']}: {pred['discount_applied']} "
                f"(confidence: {pred['confidence']:.2%})"
            )
    else:
        print(f"Error: {response.json()}")

    return response.status_code == 200


def main():
    """Ejecutar todos los tests."""
    print("\n" + "=" * 70)
    print("🧪 TESTING API DE PREDICCION DE DESCUENTOS - TESTS BASICOS")
    print("=" * 70)
    print(f"\nAPI URL: {API_URL}")
    print("\n⚠️  Asegúrate de que la API esté corriendo (python run_api.py)")

    results = []

    try:
        # Ejecutar tests
        results.append(("Health Check", test_health()))
        results.append(("Model Info", test_model_info()))
        results.append(("Single Prediction", test_single_prediction()))
        results.append(("Batch Prediction", test_batch_prediction()))

        # Resumen
        print("\n" + "=" * 70)
        print("📊 RESUMEN DE TESTS")
        print("=" * 70)

        for test_name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {test_name}: {status}")

        total_passed = sum(1 for _, passed in results if passed)
        print(f"\nTotal: {total_passed}/{len(results)} tests passed")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: No se pudo conectar a la API")
        print("   Asegúrate de que la API esté corriendo:")
        print("   python run_api.py")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    main()
