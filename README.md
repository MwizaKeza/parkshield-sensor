## Script Overview

- **main.py**: Handles server logic. Receives images from the camera page, processes them using the model, and returns JSON responses containing predictions (`label`, `confidence`, `camera_id`).
- **index.html**: Simulated on-ground camera sensor which captures images from a device camera and sends them to the FastAPI server’s `/api/detect` endpoint. Displays predictions for demonstration.
- **dashboard**: Web application
- **model/**: Contains the model `tflite_learn_820319_3.tflite` and labels `labels.txt` used for predictions.
- **requirements.txt**: Python packages

**Production:**

**Access deployed camera sensor and capture an animal image**

*(Note that if the render deployment has not been used before or has not be active for a while, it will take a while to load)*
```bash
https://sentra-93v5.onrender.com/static/index.html

```

**Retrieve model predictions from api server to display on dashboard**

*Send GET request to `/api/alerts`*
```bash
https://sentra-93v5.onrender.com/api/alerts
```

**JSON response e.g:**
```json
[
  {
    "camera": "camera_1",
    "prediction": "leopard",
    "confidence": 0.9999,
    "timestamp": "2025-11-09T19:35:00.123456"
  },
]
```

**Testing: Run locally**

```bash
python -m venv env
source env/bin/activate  # or for Windows: env\Scripts\activate
pip install -r requirements.txt
```

**Replace url in index.html**
```bash
BACKEND_URL = "https://localhost:8000/api/detect"
```

**Replace url in dashboard.html**
```bash
BACKEND_URL = "http://localhost:8000/api/alerts"
```

