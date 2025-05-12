#!/bin/bash

set -euo pipefail

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}

echo "Python version: ${py_version}"
echo "Docker socket: " $(ls /var/run/docker.sock)

nvidia-smi

# Run tests in the same container as the build
docker run --rm \
  --workdir /workspace/cccl/python/cuda_parallel \
  --mount type=bind,source=${HOST_WORKSPACE},target=/workspace/ \
  -e NVIDIA_DISABLE_REQUIRE=true \
  --gpus device=${NVIDIA_VISIBLE_DEVICES} \
  rapidsai/citestwheel:cuda12.8.0-rockylinux8-py3.10 \
  bash -c '\
    source /workspace/cccl/ci/build_common.sh && \
    print_environment_details && \
    fail_if_no_gpu && \
    CUDA_CCCL_WHEEL_PATH="$(ls /workspace/wheelhouse/cuda_cccl-*.whl)" && \
    CUDA_PARALLEL_WHEEL_PATH="$(ls /workspace/wheelhouse/cuda_parallel-*.whl)" && \
    python -m pip install "${CUDA_CCCL_WHEEL_PATH}" && \
    python -m pip install "${CUDA_PARALLEL_WHEEL_PATH}[test]" && \
    cd /workspace/cccl/python/cuda_parallel/tests/ && \
    python -m pytest -n ${PARALLEL_LEVEL} -v'
