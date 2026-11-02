from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os
import threading
import time
from pathlib import Path
try:
    from .sentence_recognizer import SentenceRecognizer
except ImportError:
    from sentence_recognizer import SentenceRecognizer

MODEL_PATH = Path(__file__).parent / "models" / "holistic_landmarker.task"
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"MediaPipe model not found at {MODEL_PATH}. "
        "Download holistic_landmarker.task as described in SETUP.md."
    )

holistic = mp.tasks.vision.HolisticLandmarker.create_from_options(
    mp.tasks.vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
)
holistic_lock = threading.Lock()
last_timestamp_ms = 0

app = FastAPI(title="MediaPipe Landmark Extractor")
sentence_recognizer = None
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(Path(__file__).parent / "client.html")

def extract_landmarks_from_frame(frame_bgr):
    global last_timestamp_ms
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    with holistic_lock:
        timestamp_ms = max(time.monotonic_ns() // 1_000_000, last_timestamp_ms + 1)
        results = holistic.detect_for_video(image, timestamp_ms)
        last_timestamp_ms = timestamp_ms

    landmarks = []
    for detected, count, include_visibility in (
        (results.pose_landmarks, 33, True),
        (results.left_hand_landmarks, 21, False),
        (results.right_hand_landmarks, 21, False),
    ):
        selected = detected[:count]
        for lm in selected:
            landmarks.extend([lm.x, lm.y, lm.z])
            if include_visibility:
                landmarks.append(lm.visibility if lm.visibility is not None else 0.0)
        landmarks.extend([0.0] * (count - len(selected)) * (4 if include_visibility else 3))
    return landmarks

@app.post("/recognize/frame")
async def recognize_frame(file: UploadFile = File(...)):
    global sentence_recognizer
    model_dir = Path(__file__).parent / "data" / "models"
    if sentence_recognizer is None:
        try:
            sentence_recognizer = SentenceRecognizer(model_dir)
        except FileNotFoundError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    image_data = await file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Uploaded frame is not a valid image.")
    return sentence_recognizer.observe(extract_landmarks_from_frame(image))

@app.post("/recognize/reset")
async def reset_recognition():
    if sentence_recognizer is not None:
        sentence_recognizer.reset()
    return {"sentence": "", "reset": True}

def extract_landmarks_from_video_path(video_path, max_frames=None, resize_width=640):
    cap = cv2.VideoCapture(video_path)
    frames_landmarks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_width:
            h, w = frame.shape[:2]
            scale = resize_width / max(w, 1)
            if scale != 1:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        lm = extract_landmarks_from_frame(frame)
        frames_landmarks.append(lm)
        if max_frames and len(frames_landmarks) >= max_frames:
            break
    cap.release()
    return np.array(frames_landmarks, dtype=np.float32)

@app.post("/extract/video")
async def extract_video(file: UploadFile = File(...), max_frames: int = 300):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_name = tmp.name
        tmp.write(await file.read())
    try:
        landmarks = extract_landmarks_from_video_path(tmp_name, max_frames=max_frames)
        return JSONResponse(content={"frames": landmarks.tolist(), "shape": landmarks.shape})
    finally:
        try:
            os.remove(tmp_name)
        except:
            pass

@app.post("/extract/image")
async def extract_image(file: UploadFile = File(...)):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error":"invalid image"})
    lm = extract_landmarks_from_frame(img)
    return JSONResponse(content={"landmarks": lm})
