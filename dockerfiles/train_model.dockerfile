# Base image
FROM --platform=linux/amd64 python:3.11-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# copy over our application (the essential parts) from our computer to the container
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_g22_2024/ mlops_g22_2024/
COPY data/ data/

# set the working directory
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# name our training script as the entrypoint for our docker image
ENTRYPOINT ["python", "-u", "mlops_g22_2024/train_model.py"]