import requests

url = "http://localhost:8000/"

file_path = "/home/mansur/Documents/Geovista/yolo_docker/images_10219.jpg"

# Abra a imagem e envie com a requisição POST
with open(file_path, "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url + "detect", files=files)

# Requisição GET para o endpoint /device-info
device_info_response = requests.get(url + "device-info").json()

# Imprima as respostas da API
print(response.json())
print(device_info_response)