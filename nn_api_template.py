# This example builds upon a FastAPI app serving a CNN model for real-time predictions
# using WebSockets. We include authentication, WebSocket support, model quantization,
# and logging via TensorBoard. Grafana requires exporting metrics to Prometheus, which is noted.

import torch
import torch.nn as nn
import torch.quantization
import torchvision.transforms as transforms
from fastapi import FastAPI, WebSocket, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.websockets import WebSocketDisconnect
from torch.utils.tensorboard import SummaryWriter
import uvicorn
import numpy as np
import secrets
import datetime

# Dummy CNN model definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 13 * 13, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# Quantize model for lower latency
model_fp32 = SimpleCNN()
model_fp32.eval()
model_quantized = torch.quantization.quantize_dynamic(
    model_fp32, {nn.Linear}, dtype=torch.qint8
)

# TensorBoard writer
writer = SummaryWriter(log_dir="./runs/cnn_inference")

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

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            image = torch.tensor(np.frombuffer(data, dtype=np.float32)).reshape(1, 1, 28, 28)
            with torch.no_grad():
                output = model_quantized(image)
                pred = output.argmax(dim=1).item()
                writer.add_scalar("Prediction/Label", pred, global_step=datetime.datetime.now().timestamp())
                await websocket.send_text(f"Prediction: {pred}")
    except WebSocketDisconnect:
        print("WebSocket disconnected")

@app.get("/secure-ping")
def secure_ping(username: str = Depends(authenticate)):
    return {"message": f"Hello {username}, you are authenticated."}

# Prometheus/Grafana note:
# You would expose a /metrics endpoint using the `prometheus_fastapi_instrumentator`
# and point Grafana to a Prometheus instance scraping from this endpoint.
# Example:
# from prometheus_fastapi_instrumentator import Instrumentator
# Instrumentator().instrument(app).expose(app)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
