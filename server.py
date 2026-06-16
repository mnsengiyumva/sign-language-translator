"""
=============================================================================
server.py — FastAPI Backend for the Sign Language Translator
=============================================================================

WHAT THIS SCRIPT DOES:
    This is the bridge between your trained AI model and the React frontend.

    The React app sends webcam frames to this server via WebSocket (a live
    two-way connection). For each frame received, the server:
        1. Decodes the image
        2. Runs MediaPipe to extract hand landmarks
        3. Feeds the landmarks into the trained TensorFlow model
        4. Sends back the predicted letter and confidence score

    WebSocket is used instead of regular HTTP requests because it keeps
    a persistent connection open — perfect for real-time video streaming
    where we need predictions on every single frame.

HOW TO RUN:
    uvicorn server:app --reload --port 8000

REQUIREMENTS:
    pip install fastapi uvicorn websockets python-multipart
=============================================================================
"""

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# =============================================================================
# LOAD MODEL AND LABEL MAP
# =============================================================================

print("Loading ASL model...")
model = tf.keras.models.load_model("models/asl_model.keras")
label_map = np.load("models/label_map.npy", allow_pickle=True)
print(f"Model loaded. {len(label_map)} classes ready.")

# =============================================================================
# MEDIAPIPE SETUP
# =============================================================================

# static_image_mode=False = video mode, faster for live streaming
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

app = FastAPI(title="Sign Language Translator API")

# CORS allows the React frontend (running on port 3000) to talk to this
# backend (running on port 8000) without being blocked by the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# HELPER: process one frame and return prediction
# =============================================================================

def process_frame(image_data: bytes):
    """
    Takes raw image bytes from the frontend, runs MediaPipe and the model
    on it, and returns a dictionary with the predicted letter, confidence,
    and whether a hand was detected.
    """

    # Decode the image bytes into a numpy array OpenCV can work with
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"letter": None, "confidence": 0, "hand_detected": False}

    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run MediaPipe hand detection
    result = hands.process(frame_rgb)

    if not result.multi_hand_landmarks:
        return {"letter": None, "confidence": 0, "hand_detected": False}

    # Extract the 63 landmark values
    hand_landmarks = result.multi_hand_landmarks[0]
    landmark_list = []
    for lm in hand_landmarks.landmark:
        landmark_list.extend([lm.x, lm.y, lm.z])

    # Run the model
    input_data = np.array(landmark_list, dtype=np.float32).reshape(1, -1)
    predictions = model.predict(input_data, verbose=0)[0]

    confidence = float(np.max(predictions))
    class_index = int(np.argmax(predictions))
    letter = label_map[class_index]

    return {
        "letter": letter,
        "confidence": round(confidence * 100, 1),
        "hand_detected": True
    }

# =============================================================================
# WEBSOCKET ENDPOINT
# =============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint that the React frontend connects to.

    The frontend sends frames as base64-encoded strings.
    This endpoint decodes each frame, runs the model, and sends
    back the prediction as a JSON string.

    The connection stays open for the entire session — no need to
    reconnect for every frame like you would with HTTP.
    """

    await websocket.accept()
    print("Frontend connected via WebSocket")

    try:
        while True:
            # Wait for the next frame from the frontend
            data = await websocket.receive_text()

            # The frontend sends frames as: "data:image/jpeg;base64,/9j/4AAQ..."
            # We strip the header and decode just the base64 image data
            if "," in data:
                base64_data = data.split(",")[1]
            else:
                base64_data = data

            image_bytes = base64.b64decode(base64_data)

            # Process the frame and get the prediction
            result = process_frame(image_bytes)

            # Send the prediction back to the frontend as JSON
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print("Frontend disconnected")

# =============================================================================
# HEALTH CHECK ENDPOINT
# =============================================================================

@app.get("/")
def health_check():
    """
    Simple endpoint to verify the server is running.
    Visit http://localhost:8000 in your browser to check.
    """
    return {"status": "running", "model": "ASL Classifier", "classes": len(label_map)}

