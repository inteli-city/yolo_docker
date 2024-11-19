import os
import cv2
import torch
import uvicorn
import logging
import traceback
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import pytz
from logging.handlers import RotatingFileHandler
import warnings

from fastapi import FastAPI, Query, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Define o timezone (exemplo com "America/Sao_Paulo")
TIMEZONE = pytz.timezone("America/Sao_Paulo")

# Função para adicionar o timezone local ao log
class TimezoneFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # Converte o tempo de UTC para o timezone especificado
        dt = datetime.fromtimestamp(record.created, TIMEZONE)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.strftime("%d:%m:%Y %H:%M:%S,%f")[:-3]  # Remover os últimos 3 dígitos dos microssegundos para milissegundos

# Configuração do logger para adicionar logs a um arquivo .log
log_formatter = TimezoneFormatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = "data/app.log"

file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

# Capturar warnings
logging.captureWarnings(True)
warnings_logger = logging.getLogger("py.warnings")
warnings_logger.addHandler(file_handler)

app = FastAPI(title="YOLO Inference API", description="An API for running YOLO inference in streaming mode, compatible with models from yolov8.", version="1.0")

# Summon a Yolo model
model = YOLO(os.environ.get('MODEL_WEIGHTS'))
model.fuse()
classes_name = model.names  # Dictionary of class names

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

@app.post("/detect_image_path")
async def detect_objects_from_path(
    image_path: str = Query(...,  description="Path to the image for prediction"),
    conf: float     = Query(0.5,  description="Confidence threshold for predictions"),
    iou: float      = Query(0.4,  description="Intersection over Union (IoU) threshold for NMS"),
    max_det: int    = Query(100,  description="Maximum number of detections per image"),
    classes: str    = Query(None, description="Optional filter by class, i.e. '0,1,2' for specific classes")
):
    try:
        logger.info("Received image path for detection")
        frame = cv2.imread(image_path)
        if frame is None:
            raise FileNotFoundError(f"Image at path {image_path} could not be found or read.")
        return await inference(frame, conf, iou, max_det, classes)
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": str(e)})

@app.post("/detect_image")
async def detect_objects_from_upload(
    image: UploadFile = File(...),
    conf: float     = Query(0.5,  description="Confidence threshold for predictions"),
    iou: float      = Query(0.4,  description="Intersection over Union (IoU) threshold for NMS"),
    max_det: int    = Query(100,  description="Maximum number of detections per image"),
    classes: str    = Query(None, description="Optional filter by class, i.e. '0,1,2' for specific classes")
):
    try:
        logger.info("Received image for detection")
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Uploaded image could not be read. It may be corrupted or in an unsupported format.")
        return await inference(frame, conf, iou, max_det, classes)
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": str(e)})

async def inference(frame,
                    conf: float,
                    iou: float,
                    max_det: int,
                    classes: str,
                    ):
    try:
        image_shape = frame.shape

        # Convert classes string to list of integers if provided
        classes_list = [int(c) for c in classes.split(',')] if classes else None

        # Inference
        logger.info(f"Starting inference with conf={conf}, iou={iou}, max_det={max_det}, classes={classes_list}")
        results = model.predict(source=frame,
                                conf=conf,
                                iou=iou,
                                max_det=max_det,
                                classes=classes_list)

        # Process results list
        detections = []
        for result in results:
            boxes     = result.boxes  # Boxes object for bounding box outputs
            masks     = result.masks  # Masks object for segmentation masks outputs
            probs     = result.probs  # Probs object for classification outputs
            keypoints = result.keypoints  # Keypoints object for pose outputs

            if len(boxes) > 0:
                # Draw bounding boxes when there is a detection
                for box, cls, conf, mask in zip(boxes.xyxyn, boxes.cls, boxes.conf, masks.xyn):  # Iterate over each detection
                    bbox = box[0:4]  # Bounding box coordinates [x_min, y_min, x_max, y_max]
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
                                                      image_shape=(image_shape[1], image_shape[0]),
                                                      mask=mask,
                                                      ))

        logger.info(f"Detection completed with {len(detections)} objects detected.")
        return JSONResponse(content=[detection.dict() for detection in detections])
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"message": str(e)})

# Nova rota para verificar qual GPU está sendo usada
@app.get("/device-info")
async def device_info():
    logger.info("Device info requested")
    return get_device_info()

# Nova rota para consultar as classes disponíveis no modelo
@app.get("/model_classes")
async def get_classes():
    logger.info("Classes info requested")
    return JSONResponse(content={"model_classes": classes_name})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None, access_log=False)
