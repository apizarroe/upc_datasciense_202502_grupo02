# API de Predicción de Descuentos

API REST construida con FastAPI para predecir si se aplicará descuento a una transacción usando el modelo entrenado.

## 🚀 Inicio Rápido

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Entrenar el modelo (si no lo has hecho)
```bash
python src/pipeline/training.py
```

### 3. Ejecutar la API
```bash
# Opción 1: Usando script (recomendado)
python run_api.py

# Opción 2: Directamente con uvicorn
uvicorn src.api.main:app --reload

# Opción 3: Producción (múltiples workers)
uvicorn src.api.main:app --workers 4
```

### 4. Acceder a la documentación
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

---

## 📋 Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Información de la API |
| GET | `/health` | Health check y estado del modelo |
| GET | `/model/info` | Información del modelo cargado |
| POST | `/predict` | Predicción individual |
| POST | `/predict/batch` | Predicción en lote |

### Ejemplos de Request/Response

<details>
<summary><b>POST /predict</b> - Predicción Individual</summary>

**Request (con todas las 28 features):**
```json
{
  "features": {
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
    "Payment Method_2": 1
  }
}
```

**Response:**
```json
{
  "discount_applied": true,
  "prediction": 1,
  "probability_discount": 0.7655,
  "probability_no_discount": 0.2345,
  "confidence": 0.7655,
  "timestamp": "2025-10-17T10:30:00.123456"
}
```
</details>

<details>
<summary><b>POST /predict/batch</b> - Predicción Batch</summary>

**Request (con todas las 28 features):**
```json
{
  "data_list": [
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
      "Payment Method_2": 1
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
      "Payment Method_2": 0
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "index": 0,
      "discount_applied": true,
      "prediction": 1,
      "confidence": 0.7655
    },
    {
      "index": 1,
      "discount_applied": false,
      "prediction": 0,
      "confidence": 0.8234
    }
  ],
  "total": 2,
  "discount_count": 1,
  "no_discount_count": 1
}
```
</details>

---

## 💻 Ejemplos de Uso

### Python
```python
import requests

API_URL = "http://localhost:8000"

# Predicción individual
response = requests.post(
    f"{API_URL}/predict",
    json={
        "features": {
            "Price Per Unit": -0.453,
            "Quantity": 1.592,
            "Customer_Transaction_Count": 0.726,
            "Transaction_Year": 2024,
            "Category_7": 1,
            "Location_1": 1
        }
    }
)

result = response.json()
print(f"Descuento: {result['discount_applied']}")
print(f"Confianza: {result['confidence']:.2%}")
```

### cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {"Price Per Unit": -0.453, "Quantity": 1.592, "Transaction_Year": 2024}}'
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    features: {
      'Price Per Unit': -0.453,
      'Quantity': 1.592,
      'Transaction_Year': 2024,
      'Category_7': 1
    }
  })
});

const result = await response.json();
console.log('Descuento:', result.discount_applied);
```

---

## 🧪 Tests

```bash
# Tests básicos
python src/api/tests/test_basic.py

# Tests con datos reales
python src/api/tests/test_with_real_data.py
```

Ver más detalles en [tests/README.md](tests/README.md)

---

## ⚙️ Estructura

```
src/api/
├── main.py              # API FastAPI
├── schemas.py           # Modelos Pydantic
├── README.md            # Esta documentación
└── tests/
    ├── test_basic.py
    └── test_with_real_data.py
```

**Integración con pipeline:**
- `src/pipeline/predict.py` - Funciones de predicción
- `src/pipeline/training.py` - Entrenamiento del modelo

---

## 🔧 Troubleshooting

| Error | Solución |
|-------|----------|
| "Modelo no disponible" | `python src/pipeline/training.py` |
| "No module named 'fastapi'" | `pip install -r requirements.txt` |
| Puerto 8000 ocupado | `uvicorn src.api.main:app --port 8001` |
| Features faltantes | Consulta `/model/info` para ver features requeridas |

---

## 📝 Notas Importantes

### Features Requeridas (28 en total)

El modelo espera **28 features preprocesadas**:

**Numéricas (2):**
- `Price Per Unit`, `Quantity`

**Features de Cliente (4):**
- `Customer_Transaction_Count`, `Customer_Avg_Spent`
- `Customer_Discount_Frequency`, `Customer_Recency`

**Features de Producto (4):**
- `Product_Transaction_Count`, `Product_Avg_Price`
- `Product_Discount_Rate`, `Product_Popularity_Score`

**Features Temporales (4):**
- `Transaction_Year`, `Transaction_Month`
- `Transaction_Quarter`, `Transaction_DayOfWeek`

**Interacciones (1):**
- `Price_Quantity_Interaction`

**One-Hot Encoded (13):**
- `Category_0` a `Category_7` (8 categorías)
- `Location_0` a `Location_1` (2 ubicaciones)
- `Payment Method_0` a `Payment Method_2` (3 métodos de pago)

### Comportamiento

- **Features**: Deben estar preprocesadas (escaladas, codificadas como en `data/processed/`)
- **Features faltantes**: Se rellenan automáticamente con 0 (con warning en logs)
- **Orden**: El orden de las features se ajusta automáticamente
- **CORS**: Habilitado para desarrollo (configurar para producción)
- **Códigos HTTP**: 200 (OK), 400 (Bad Request), 503 (No disponible), 500 (Error)
