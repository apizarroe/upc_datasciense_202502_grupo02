"""Tests basicos para la API de prediccion de descuentos."""

import json

import requests

# URL de la API
API_URL = "http://localhost:8000"


def test_health() -> bool:
    """Test del endpoint de health check."""
    print("\n" + "=" * 70)
    print("1️⃣  TEST: Health Check")
    print("=" * 70)

    response = requests.get(f"{API_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response:\n{json.dumps(response.json(), indent=2)}")

    return response.status_code == 200


def test_model_info() -> bool:
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
        print("\nFirst 10 Features:")
        for i, feature in enumerate(data["feature_columns"], 1):
            print(f"  {i}. {feature}")
    else:
        print(f"Error: {response.json()}")

    return response.status_code == 200


def test_single_prediction() -> bool:
    """Test del endpoint de prediccion individual."""
    print("\n" + "=" * 70)
    print("3️⃣  TEST: Single Prediction")
    print("=" * 70)

    # Ejemplo con TODAS las 28 features del modelo
    sample_features = {
        "Price Per Unit": -0.453,
        "Quantity": 1.592,
        "Customer_Transaction_Count": 0.726,
        "Customer_Avg_Spent": -1.755,
        "Customer_Discount_Frequency": -0.292,
        "Customer_Recency": -0.267,
        "Product_Transaction_Count": -1.642,
        "Product_Avg_Price": -0.453,
        "Product_Discount_Rate": -0.086,
        "Product_Popularity_Score": -1.642,
        "Transaction_Year": 2024,
        "Transaction_Month": 4,
        "Transaction_Quarter": 2,
        "Transaction_DayOfWeek": 0,
        "Price_Quantity_Interaction": 0.586,
        "Category_0": 0,
        "Category_1": 0,
        "Category_2": 0,
        "Category_3": 0,
        "Category_4": 0,
        "Category_5": 0,
        "Category_6": 0,
        "Category_7": 1,
        "Location_0": 0,
        "Location_1": 1,
        "Payment Method_0": 0,
        "Payment Method_1": 0,
        "Payment Method_2": 1,
    }

    print("\nRequest Features:")
    print(json.dumps(sample_features, indent=2))

    response = requests.post(
        f"{API_URL}/predict", json={"features": sample_features}
    )

    print(f"\nStatus Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("\nPrediction Results:")
        print(f"  Discount Applied: {result['discount_applied']}")
        print(f"  Prediction: {result['prediction']}")
        prob_no_discount = result["probability_no_discount"]
        print(f"  Probability No Discount: {prob_no_discount:.4f}")
        print(f"  Probability Discount: {result['probability_discount']:.4f}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print(f"  Timestamp: {result['timestamp']}")
    else:
        print(f"Error: {response.json()}")

    return response.status_code == 200


def test_batch_prediction() -> bool:
    """Test del endpoint de prediccion batch."""
    print("\n" + "=" * 70)
    print("4️⃣  TEST: Batch Prediction")
    print("=" * 70)

    # Ejemplo de multiples transacciones con TODAS las 28 features
    batch_data = [
        {
            "Price Per Unit": -0.453,
            "Quantity": 1.592,
            "Customer_Transaction_Count": 0.726,
            "Customer_Avg_Spent": -1.755,
            "Customer_Discount_Frequency": -0.292,
            "Customer_Recency": -0.267,
            "Product_Transaction_Count": -1.642,
            "Product_Avg_Price": -0.453,
            "Product_Discount_Rate": -0.086,
            "Product_Popularity_Score": -1.642,
            "Transaction_Year": 2024,
            "Transaction_Month": 4,
            "Transaction_Quarter": 2,
            "Transaction_DayOfWeek": 0,
            "Price_Quantity_Interaction": 0.586,
            "Category_0": 0,
            "Category_1": 0,
            "Category_2": 0,
            "Category_3": 0,
            "Category_4": 0,
            "Category_5": 0,
            "Category_6": 0,
            "Category_7": 1,
            "Location_0": 0,
            "Location_1": 1,
            "Payment Method_0": 0,
            "Payment Method_1": 0,
            "Payment Method_2": 1,
        },
        {
            "Price Per Unit": 0.234,
            "Quantity": -0.567,
            "Customer_Transaction_Count": -1.234,
            "Customer_Avg_Spent": 0.891,
            "Customer_Discount_Frequency": 0.456,
            "Customer_Recency": 0.123,
            "Product_Transaction_Count": 0.789,
            "Product_Avg_Price": 0.234,
            "Product_Discount_Rate": 0.345,
            "Product_Popularity_Score": 0.789,
            "Transaction_Year": 2023,
            "Transaction_Month": 8,
            "Transaction_Quarter": 3,
            "Transaction_DayOfWeek": 2,
            "Price_Quantity_Interaction": -0.234,
            "Category_0": 1,
            "Category_1": 0,
            "Category_2": 0,
            "Category_3": 0,
            "Category_4": 0,
            "Category_5": 0,
            "Category_6": 0,
            "Category_7": 0,
            "Location_0": 1,
            "Location_1": 0,
            "Payment Method_0": 1,
            "Payment Method_1": 0,
            "Payment Method_2": 0,
        },
    ]

    print(f"\nRequest: {len(batch_data)} items")

    response = requests.post(
        f"{API_URL}/predict/batch", json={"data_list": batch_data}
    )

    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        result = response.json()
        print("\nBatch Prediction Results:")
        print(f"  Total: {result['total']}")
        print(f"  Discount Count: {result['discount_count']}")
        print(f"  No Discount Count: {result['no_discount_count']}")

        print("\n  Individual Predictions:")
        for pred in result["predictions"]:
            print(
                f"    #{pred['index']}: {pred['discount_applied']} "
                f"(confidence: {pred['confidence']:.2%})"
            )
    else:
        print(f"Error: {response.json()}")

    return response.status_code == 200


def main() -> None:
    """Ejecutar todos los tests."""
    print("\n" + "=" * 70)
    print("🧪 TESTING API DE PREDICCION DE DESCUENTOS - TESTS BASICOS")
    print("=" * 70)
    print(f"\nAPI URL: {API_URL}")
    msg = "\n⚠️  Asegúrate de que la API esté corriendo (python run_api.py)"
    print(msg)

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
