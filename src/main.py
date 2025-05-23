# This example builds upon a FastAPI app serving a CNN model for real-time predictions
# using WebSockets. We include authentication, WebSocket support, model quantization,
# and logging via TensorBoard. Grafana requires exporting metrics to Prometheus, which is noted.

import torch
import torch.nn as nn
import torch.quantization
import torchvision.transforms as transforms
from fastapi.responses import HTMLResponse
from fastapi import FastAPI, WebSocket, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.websockets import WebSocketDisconnect
from prometheus_fastapi_instrumentator import Instrumentator
from torch.utils.tensorboard import SummaryWriter
import uvicorn
import numpy as np
import secrets
import datetime
from prometheus_client import Summary, Gauge, Counter
import time

PREDICTION_LATENCY = Summary('prediction_latency_seconds', 'Time spent in prediction')
MODEL_CONFIDENCE = Gauge('model_confidence', 'Confidence score of latest prediction')
PREDICTION_COUNT = Counter('prediction_requests_total', 'Total number of predictions')

# model definition
from models.hybrid_cnn_lstm import EcogToMotionNet


# Quantize model for lower latency
model_name = "./src/models/Hybrid_CNN_LSTM_ipsilateral_3_output.pth"
model = EcogToMotionNet()
model.load_state_dict(torch.load(model_name))

# model_fp32 = EcogToMotionNet()
# model_fp32.eval()
# model_quantized = torch.quantization.quantize_dynamic(
#     model_fp32, {nn.Linear}, dtype=torch.qint8
# )

# TensorBoard writer
writer = SummaryWriter(log_dir="./runs/model_inference")

# FastAPI app
app = FastAPI()
security = HTTPBasic()
USERNAME = "admin"
PASSWORD = "secure123"

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# Prometheus/Grafana note:
# You would expose a /metrics endpoint using the `prometheus_fastapi_instrumentator`
# and point Grafana to a Prometheus instance scraping from this endpoint.
# Example:
# from prometheus_fastapi_instrumentator import Instrumentator
# Instrumentator().instrument(app).expose(app)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h1>API running. Metrics available at <a href='/metrics'>/metrics</a></h1>"

@app.get("/healthz")
def health_check():
    return {"status": "ok"}

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            # Deserialize input to numpy array
            input_array = np.frombuffer(data, dtype=np.float32)
            # Assuming input shape known, e.g., (N,), adjust reshape accordingly if needed
            # For example, flatten input or reshape for your model
            input_tensor = torch.tensor(input_array).unsqueeze(0)  # batch dimension

            start_time = time.time()
            with torch.no_grad():
                output = model(input_tensor)  # Expected shape (1, 6)
            latency = time.time() - start_time

            coords = output.squeeze(0).numpy()  # shape (6,)

            # Prometheus metrics update
            PREDICTION_LATENCY.observe(latency)
            PREDICTION_COUNT.inc()

            # TensorBoard logging
            timestamp = int(datetime.datetime.now().timestamp())
            writer.add_scalar("Prediction/Latency", latency, global_step=timestamp)
            # Log each coordinate separately
            for i, coord in enumerate(coords):
                writer.add_scalar(f"Prediction/Coordinate_{i}", coord, global_step=timestamp)

            # Send coordinates and confidence back as JSON string
            response = {
                "coordinates": coords.tolist(),
            }
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("WebSocket disconnected")

@app.get("/secure-ping")
def secure_ping(username: str = Depends(authenticate)):
    return {"message": f"Hello {username}, you are authenticated."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
