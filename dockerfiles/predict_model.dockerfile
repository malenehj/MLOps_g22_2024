#Base image
FROM --platform=linux/amd64 python:3.11-slim

EXPOSE $PORT

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

CMD exec uvicorn mlops_g22_2024.fast_api:app --port $PORT --host 0.0.0.0 --workers 1