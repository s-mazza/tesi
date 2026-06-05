FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS build

ARG LLAMA_CPP_REF=master
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libcurl4-openssl-dev \
    libssl-dev \
  && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/local/cuda-12.4/compat/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so \
  && ln -sf /usr/local/cuda-12.4/compat/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1
ENV LIBRARY_PATH=/usr/local/cuda-12.4/compat:/usr/local/cuda-12.4/targets/x86_64-linux/lib/stubs

WORKDIR /src/llama.cpp
RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" https://github.com/ggml-org/llama.cpp.git .
RUN cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release \
  && cmake --build build --target llama-server -j"$(nproc)"

FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates libcurl4 libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/llama.cpp/build/bin/llama-server /app/llama-server

EXPOSE 8080
ENTRYPOINT ["/app/llama-server"]
