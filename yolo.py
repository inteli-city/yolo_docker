import cv2
import ast
import math
import random
import colorsys
import argparse
import numpy as np
from PIL import Image
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--weights',     type = str,   default = "/home/mansur/Documents/Geovista/stereo_vision/ultralytics_stereo/weights/yolov8x-seg.pt", help="path dos pesos treinados")
parser.add_argument('--classes',     nargs = '+',     type = int,  help='filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument('--conf_thres',  type = float, default = 0.40, help='confidence threshold')
parser.add_argument('--iou_thres',   type = float, default = 0.40, help='NMS IoU threshold')
parser.add_argument('--max_det',     type = int,   default =    7, help='maximum detections per image')
parser.add_argument('--source',      type = str,   default =    0, help="path de video ou camera_ID")

#parser.add_argument('--meta_dados', type = str, default=False,     help='Path para corrigir perspectiva a partir de um arquivo especifico de metadados, ex: "meta-dados-dataset-cam0-20230710-11h45m46s.txt"')
args = parser.parse_args()

#cap0 = cv2.imread('/home/mansur/Documents/Geovista/yolov5-pc/perspective-transform/Screenshot from 2022-09-14 17-38-53.png')
if args.source.isdigit():

    cap0 = cv2.VideoCapture(int(args.source))

else:

    cap0 = cv2.VideoCapture((args.source))


# ajuste de resolução
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)#Ajuste de resolução nativa da logitech C270
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)#Ajuste de resolução nativa da logitech C270
width  = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))

cam0 = "yolov5 cam 0"
#cam1 = "yolov5 cam 1"

# summon a Yolov8 model
model = YOLO(args.weights)
model.fuse()
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam

classes_name = model.names      # Dicionario das classes

# Gera as cores para as classes
def get_colors(num_colors):
    # Gera uma lista de cores em formato RGB usando a biblioteca colorsys
    colors = []
    for i in range(num_colors):
        hue = i / float(num_colors)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors

colors = get_colors(len(classes_name))

while cap0.isOpened():
    
    ret, frame0 = cap0.read()
    
    # Inference
    results = model.predict(source      = frame0, 
                            iou         = args.iou_thres,  # NMS IoU (Intersection over Union) threshold
                            conf        = args.conf_thres, # NMS confidence threshold
                            max_det     = args.max_det,    # maximum number of detections per image
                            classes     = args.classes)    #[1, 2, 16]  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs

    # Process results list
    for result in results:
        boxes     = result.boxes            # Boxes object for bounding box outputs
        masks     = result.masks            # Masks object for segmentation masks outputs
        probs     = result.probs            # Probs object for classification outputs
        keypoints = result.keypoints        # Keypoints object for pose outputs
        #result.show()                      # display to screen
        #result.save(filename='result.jpg') # save to disk
    #annoted_frame = results[0].plot()
        
    img = frame0.copy()
    detections = []

    if len(boxes) > 0:
        # Desenhar bounding boxes quando há detecção
        for box, cls, conf, mask in zip(boxes.xyxy, boxes.cls, boxes.conf, masks.xy):  # Iterar sobre cada detecção
            bbox = box[0:4]  # Coordenadas da bounding box [x_min, y_min, x_max, y_max]
            bbox = [int(coord) for coord in bbox] # Converter coordenadas para inteiros
            cls = int(cls.item()) # extraindo dados de um tensor
            label = classes_name[cls]  # Rótulo da classe
            thickness = 2
            
            # Exibir boundingbox com classe e confiança
            labels = f"Label: {cls} {label}, Conf: {conf:.2f} x{bbox[0]} y{bbox[1]} w{bbox[2]-bbox[0]} h{bbox[3]-bbox[1]}"
            detections.append(label)

            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[cls], thickness)
            cv2.putText(img, labels, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[cls], thickness)
            cv2.fillPoly(img, np.int32([mask]), colors[cls])
        
        # Mix image and masks
        img = cv2.addWeighted(frame0, 0.7, img, 0.3, 0)

    #cv2.namedWindow (str(cam0), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
    #cv2.imshow(str(cam0), img)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        message = b'bye'
        cap0.release()
        cv2.destroyAllWindows()
        break