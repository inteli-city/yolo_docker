import requests

url = "http://localhost:8000/"

# Lista de caminhos para as imagens
image_paths = [
    "/home/mansur/Documents/Geovista/yolo_docker/images_10219.jpg",
    "/home/mansur/Documents/Geovista/yolo_docker/images_3716.jpg",
    "/home/mansur/Documents/Geovista/yolo_docker/images_36991.jpg"
]

# Loop para enviar cada imagem para a API
for file_path in image_paths:
    with open(file_path, "rb") as image_file:
        files = {"image": image_file}
        response = requests.post(url + "detect", files=files)
        print(f"Response for {file_path}: {response.json()}")

# Requisição GET para o endpoint /device-info
device_info_response = requests.get(url + "device-info").json()
print(device_info_response)
