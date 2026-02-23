import cv2
import numpy as np
import dlib
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = dlib.get_frontal_face_detector()

@app.post("/headpose")
async def headpose(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return {"detected": False}

    face = faces[0]

    # Bounding box center
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    h, w = frame.shape[:2]

    # Normalize to -0.5 to +0.5 range
    norm_x = (center_x / w) - 0.5
    norm_y = (center_y / h) - 0.5

    # Scale to small stable values
    norm_x *= 0.5
    norm_y *= 0.5

    return {
        "detected": True,
        "tx": float(norm_x),
        "ty": float(norm_y)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)