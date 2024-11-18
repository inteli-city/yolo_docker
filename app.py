import os
import cv2
import torch
import uvicorn
import colorsys
import traceback
import numpy as np
from ultralytics import YOLO

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="YOLO Inference API", description="An API for running YOLO inference in streaming mode.", version="1.0")

# Summon a Yolo model
#model = YOLO("/data/best.pt")
model = YOLO(os.environ.get('MODEL_WEIGHTS'))
model.fuse()
classes_name = model.names  # Dictionary of class names

# Generate colors for the classes
def get_colors(num_colors):
    colors = []
    for i in range(num_colors):
        hue = i / float(num_colors)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

colors = get_colors(len(classes_name))

# Função para verificar o dispositivo de GPU
def get_device_info():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        devices = []
        for i in range(device_count):
            devices.append({
                "device_id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated": torch.cuda.memory_allocated(i),
                "memory_reserved": torch.cuda.memory_reserved(i)
            })
        return {"status": "GPU", "devices": devices}
    else:
        return {"status": "CPU", "devices": None}

class DetectionResult(BaseModel):
    label: str
    id: int
    confidence: float
    x: float
    y: float
    width: float
    height: float
    image_shape: list
    mask: list

@app.post("/detect")
async def detect_objects(image: UploadFile = File(...)):
    try:
        # Read image file
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        image_shape = frame.shape

        # Inference
        results = model.predict(source=frame,
                                conf=0.5,
                                iou=0.4,
                                max_det=100,
                                classes=None)

        # Process results list
        detections = []
        for result in results:
            boxes     = result.boxes  # Boxes object for bounding box outputs
            masks     = result.masks  # Masks object for segmentation masks outputs
            probs     = result.probs  # Probs object for classification outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs

            #img = frame.copy()

            if len(boxes) > 0:
                # Draw bounding boxes when there is a detection
                for box, cls, conf, mask in zip(boxes.xyxyn, boxes.cls, boxes.conf, masks.xyn):  # Iterate over each detection
                    bbox = box[0:4]  # Bounding box coordinates [x_min, y_min, x_max, y_max]
                    #bbox = [int(coord) for coord in bbox]  # Convert coordinates to integers
                    cls = int(cls.item())  # Extract data from tensor
                    label = classes_name[cls]  # Class label
                    labels = f"Label: {cls} {label}, Conf: {conf:.2f} x{bbox[0]} y{bbox[1]} w{bbox[2]-bbox[0]} h{bbox[3]-bbox[1]}"
                    mask = mask.tolist()
                    detections.append(DetectionResult(label=label,
                                                      id=cls,
                                                      confidence=conf,
                                                      x=bbox[0],
                                                      y=bbox[1],
                                                      width=(bbox[2] - bbox[0]),
                                                      height=(bbox[3] - bbox[1]),
                                                      image_shape=(image_shape[1],image_shape[0]),
                                                      mask=mask,
                                                      ))

        return JSONResponse(content=[detection.dict() for detection in detections])
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": str(e)})
    
# Nova rota para verificar qual GPU está sendo usada
@app.get("/device-info")
async def device_info():
    return get_device_info()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Dockerfile
# FROM python:3.9
# WORKDIR /app
# COPY . /app
# RUN pip install fastapi uvicorn opencv-python-headless ultralytics
# EXPOSE 8000
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
