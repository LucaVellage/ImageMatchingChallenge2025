FROM colmap/colmap:latest

ENV DEBIAN_FRONTEND=noninteractive
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY docker/colmap_jupyter/requirements_pip.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir \
        --extra-index-url "$TORCH_INDEX_URL" \
        -r /tmp/requirements.txt \
        jupyterlab

ENV PATH="/opt/venv/bin:${PATH}"

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
