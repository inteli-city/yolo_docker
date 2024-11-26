# Dockerfile
#FROM ultralytics/ultralytics
#FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:12.6.2-cudnn-runtime-ubuntu22.04

# Instalar Python e outras dependências essenciais do sistema
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Definir Python como python3
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Define o diretório de trabalho
WORKDIR /app

# Copia o código da aplicação para o diretório de trabalho
COPY . /app

# Instala dependências adicionais necessárias
#RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

# Instala as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Expõe a porta 8000 para comunicação
EXPOSE 8000

# Comando para iniciar o servidor
#CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Comando para iniciar o servidor, redirecionando logs para o arquivo e mantendo saída no terminal
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8000 2>&1 | tee -a data/app.log"]
