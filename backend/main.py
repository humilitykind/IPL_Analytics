from fastapi import FastAPI
import logging
from prometheus_fastapi_instrumentator import Instrumentator

# Step 4 of Project Spec: Implement comprehensive logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("IPL_Backend")

app = FastAPI(title="IPL Prediction Engine Backend")

# Instrument the FastAPI app to expose Prometheus metrics at /metrics
Instrumentator().instrument(app).expose(app)

@app.get("/health")
def health_check():
    """Endpoint for automated orchestration health checks."""
    logger.info("Health check endpoint called.")
    return {"status": "ok"}

@app.get("/ready")
def readiness_check():
    """Endpoint to check if the app is fully ready to accept traffic (e.g., model is loaded)."""
    logger.info("Ready check endpoint called.")
    # In the future, this will check if our ML model is successfully loaded
    return {"status": "ready"}
