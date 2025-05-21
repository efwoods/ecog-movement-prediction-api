# main.py
from fastapi import FastAPI, WebSocket, Request, Depends, status, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn
import torch
import torch.nn as nn
import torch.quantization
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import secrets

app = FastAPI()

# Authentication setup
security = HTTPBasic()
USERNAME = "admin"
PASSWORD = "password"

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, USERNAME)
    correct_password = secrets.compare_digest(credentials.password, PASSWORD)
    if not (correct_username and correct_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return credentials.username

# Logging with TensorBoard
writer = SummaryWriter(log_dir="./runs")

# Dummy model and quantization
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(5408, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

model = SimpleCNN()
model.eval()
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
model_fp32_prepared = torch.quantization.prepare(model)
model_int8 = torch.quantization.convert(model_fp32_prepared)

# Prometheus instrumentation
Instrumentator().instrument(app).expose(app)

@app.get("/", response_class=HTMLResponse)
async def root():
    return "<h1>API running. Metrics available at <a href='/metrics'>/metrics</a></h1>"

@app.get("/predict")
async def predict(username: str = Depends(get_current_username)):
    # Simulate prediction
    dummy_input = torch.rand(1, 1, 28, 28)
    output = model_int8(dummy_input)
    prediction = torch.argmax(output, dim=1).item()
    writer.add_scalar("Prediction", prediction, time.time())
    return {"prediction": prediction}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        dummy_input = torch.rand(1, 1, 28, 28)
        output = model_int8(dummy_input)
        prediction = torch.argmax(output, dim=1).item()
        writer.add_scalar("WS Prediction", prediction, time.time())
        await websocket.send_json({"prediction": prediction})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
