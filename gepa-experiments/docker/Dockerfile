ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG VLLM_PACKAGE=vllm
ARG FLASH_ATTN_WHEEL_URL=

ENV DEBIAN_FRONTEND=noninteractive
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
ENV HF_HOME=/llms
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

COPY gepa-experiments/docker/requirements.txt /tmp/geval-gepa-requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "${VLLM_PACKAGE}" \
    && python -m pip install -r /tmp/geval-gepa-requirements.txt \
    && if [ -n "${FLASH_ATTN_WHEEL_URL}" ]; then \
        python -m pip install "${FLASH_ATTN_WHEEL_URL}"; \
    else \
        echo "FLASH_ATTN_WHEEL_URL not set; relying on vLLM bundled attention backend."; \
    fi

WORKDIR /workspace
