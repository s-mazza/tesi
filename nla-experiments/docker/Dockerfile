ARG BASE_IMAGE=emb-inversion:latest
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY requirements-nla-transformers.txt /tmp/requirements-nla-transformers.txt

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install -r /tmp/requirements-nla-transformers.txt
