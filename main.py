from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import datetime
from collections import deque
import uuid
import shutil

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model/tflite_learn_820319_3.tflite"
LABELS_PATH = "model/labels.txt"

# Load model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Load labels
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        LABELS = [line.strip() for line in f.readlines() if line.strip()]
else:
    LABELS = []

# Store alerts in memory
ALERTS = deque(maxlen=50)
os.makedirs("static/alerts", exist_ok=True)

def get_danger_level(label: str):
    label = label.lower()
    if any(animal in label for animal in ["lion", "elephant", "leopard", "buffalo"]):
        return "Severely Dangerous"
    elif any(animal in label for animal in ["giraffe", "zebra"]):
        return "Moderately Urgent"
    else:
        return "Less Urgent"


@app.post("/api/detect")
async def detect(
    request: Request,  # Add Request to get the base URL
    image: UploadFile = File(...),
    camera_id: str = Form("camera_1"),
    timestamp: str = Form(None),
    latitude: str = Form(None),
    longitude: str = Form(None)
):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_resized = img.resize((128, 128))
    input_data = np.expand_dims(np.array(img_resized, dtype=np.float32) / 255.0, axis=0)

    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_index)[0]

    pred_idx = int(np.argmax(output_data))
    confidence = float(np.max(output_data))
    label = LABELS[pred_idx] if LABELS and pred_idx < len(LABELS) else f"class_{pred_idx}"

    # Save image
    image_filename = f"{uuid.uuid4().hex}.jpg"
    image_path = os.path.join("static", "alerts", image_filename)
    with open(image_path, "wb") as f:
        f.write(contents)

    danger_level = get_danger_level(label)

    # Build absolute URL for image
    base_url = str(request.base_url).rstrip("/")  # e.g., http://localhost:8000
    image_url = f"{base_url}/static/alerts/{image_filename}"

    alert = {
        "id": uuid.uuid4().hex,
        "camera": camera_id,
        "prediction": label,
        "confidence": confidence,
        "timestamp": timestamp or datetime.datetime.now().isoformat(),
        "latitude": latitude,
        "longitude": longitude,
        "status": "Pending",
        "danger_level": danger_level,
        "image_url": image_url
    }

    ALERTS.appendleft(alert)
    return {"prediction": label, "confidence": confidence, "alert": alert}

@app.get("/api/alerts")
async def get_alerts():
    return list(ALERTS)

@app.post("/api/approve/{alert_id}")
async def approve_alert(alert_id: str):
    for alert in ALERTS:
        if alert["id"] == alert_id:
            alert["status"] = "Approved"
            break
    return {"message": "Alert approved", "alert_id": alert_id}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)

