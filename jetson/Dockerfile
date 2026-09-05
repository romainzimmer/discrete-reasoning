# Jetson (JetPack 7.2.1 / L4T r39.2.1)
#
# From jetson/ (requires nvidia-container-toolkit on the host):
#   docker compose build
#   docker compose run --rm download
#   docker compose run --rm train --min-rating 0 --max-rating 0 --max-samples 100 --epochs 30
#   docker compose up viz

ARG BASE_IMAGE=whitesscott/l4t-jetpack:r39.2.1
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN mkdir -p src

# Install dependencies from pyproject.toml (cached until pyproject.toml changes).
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu132
RUN pip3 install --no-cache-dir --break-system-packages --default-timeout=86400 \
    . \
    --index-url ${TORCH_INDEX} \
    --extra-index-url https://pypi.org/simple

COPY src ./src
RUN pip3 install --no-cache-dir --break-system-packages --no-deps .

CMD ["train", "--help"]
