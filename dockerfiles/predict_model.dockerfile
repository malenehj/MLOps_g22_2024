#Base image
FROM --platform=linux/amd64 python:3.11-slim

EXPOSE 8500

WORKDIR /

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# copy over our application (the essential parts)
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_g22_2024/ mlops_g22_2024/
COPY models/ models/

# installing requirements
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mlops_g22_2024/fast_api.py", "--server.port=8500", "--server.address=0.0.0.0"]


# Base image
#FROM python:3.11-slim

#RUN apt update && \
#    apt install --no-install-recommends -y build-essential gcc && \
#    apt clean && rm -rf /var/lib/apt/lists/*

#COPY requirements.txt requirements.txt
#COPY pyproject.toml pyproject.toml
#COPY mlops_g22_2024/ mlops_g22_2024/
#COPY data/ data/

#WORKDIR /
#RUN pip install -r requirements.txt --no-cache-dir
#RUN pip install . --no-deps --no-cache-dir

#ENTRYPOINT ["python", "-u", "mlops_g22_2024/predict_model.py"]