FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-nla-experiments.txt /tmp/requirements-nla-experiments.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements-nla-experiments.txt
