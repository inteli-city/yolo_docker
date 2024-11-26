import requests

url = "http://localhost:8000/"


# Lista de caminhos para as imagens, enviando somante p path string
# Lista de caminhos para as imagens, enviando imagens
image_paths = [
    #"/home/mansur/Documents/Geovista/yolo_docker/images_10219.jpg",
    #"/home/mansur/Documents/Geovista/yolo_docker/images_3716.jpg",
    #"/home/mansur/Documents/Geovista/yolo_docker/images_36991.jpg",
    "/workspace/production/docker/AirFlow/volume/data/out/2024/06/JETSON_000108_cam0-20240605-14h28m24s.jpg"
]

# Loop para enviar cada imagem para a API
for file_path in image_paths:
    with open(file_path, "rb") as image_file:
        files = {"image": image_file}
        response = requests.post(url + "/detect_image?conf=0.5&iou=0.4&max_det=10&classes=0,1,2,3,4,7", files=files)
        print(f"Response for {file_path}: {response.json()}")

# Requisição GET para o endpoint /device-info
device_info_response = requests.get(url + "device-info").json()
print(device_info_response)


"""
"""
image_paths = [
    "./images_10219.jpg",
    "./images_3716.jpg",
    "./images_36991.jpg",
    "/datalake/images_5962.jpg",
    "/datalake/images_1332.jpg",
    "./I1_000158.png",
    "/datalake/out/2024/06/JETSON_000108_cam0-20240605-14h28m24s.jpg"
]

# Loop para enviar cada imagem para a API
for image_path in image_paths:
    params = {
        "image_path": image_path,
        "conf": 0.5,
        "iou": 0.4,
        "max_det": 10,
        "classes": "0,1,2,7" # ou "" para não filtrar
    }
    response = requests.post(url + "detect_image_path", params=params)
    print(f"Response for {image_path}: {response.json()}")

# Requisição GET para o endpoint /device-info
device_info_response = requests.get(url + "device-info").json()
print(device_info_response)