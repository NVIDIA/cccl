# syntax=docker/dockerfile:1.5

ARG BASE
ARG PYTHON_PACKAGE_MANAGER=conda

FROM ${BASE} as pip-base

RUN apt update -y \
 && DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
    # C++ build tools
    doxygen \
    graphviz \
    # C++ test dependencies
    libgmock-dev \
    libgtest-dev \
    # needed by libcudf_kafka
    librdkafka-dev \
    # cuML/cuGraph dependencies
    libblas-dev \
    liblapack-dev \
    # needed by libcuspatial
    libgdal-dev \
    sqlite3 \
    libsqlite3-dev \
    libtiff-dev \
    libcurl4-openssl-dev \
 && rm -rf /tmp/* /var/tmp/* /var/cache/apt/* /var/lib/apt/lists/*;

ENV DEFAULT_VIRTUAL_ENV=rapids

FROM ${BASE} as conda-base

ENV DEFAULT_CONDA_ENV=rapids

FROM ${PYTHON_PACKAGE_MANAGER}-base

ARG CUDA
ENV CUDAARCHS="70-real"
ENV CUDA_VERSION="${CUDA_VERSION:-${CUDA}}"

ARG PYTHON_PACKAGE_MANAGER
ENV PYTHON_PACKAGE_MANAGER="${PYTHON_PACKAGE_MANAGER}"

ENV PYTHONSAFEPATH="1"
ENV PYTHONUNBUFFERED="1"
ENV PYTHONDONTWRITEBYTECODE="1"

ENV SCCACHE_REGION="us-east-2"
ENV SCCACHE_BUCKET="rapids-sccache-devs"
ENV VAULT_HOST="https://vault.ops.k8s.rapids.ai"
ENV HISTFILE="/home/coder/.cache/._bash_history"

ENV LIBCUDF_KERNEL_CACHE_PATH="/home/coder/cudf/cpp/build/${PYTHON_PACKAGE_MANAGER}/cuda-${CUDA}/latest/jitify_cache"
