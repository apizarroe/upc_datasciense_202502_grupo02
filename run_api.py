"""Script para ejecutar la API de prediccion de descuentos."""

import uvicorn

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 INICIANDO API DE PREDICCION DE DESCUENTOS")
    print("=" * 70)
    print("\n📍 Servidor corriendo en: http://localhost:8000")
    print("📚 Documentacion Swagger: http://localhost:8000/docs")
    print("📚 Documentacion ReDoc: http://localhost:8000/redoc")
    print("❤️  Health Check: http://localhost:8000/health")
    print("\n⌨️  Presiona CTRL+C para detener el servidor\n")
    print("=" * 70)

    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
