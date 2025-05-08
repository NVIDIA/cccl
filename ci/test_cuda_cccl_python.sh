#!/bin/bash

set -euo pipefail

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}
echo "Python version: ${py_version}"
echo "Docker socket: " $(ls /var/run/docker.sock)

nvidia-smi

# Run tests in the same container as the build
docker run --rm \
  --workdir /home/coder/cccl/python/cuda_cccl \
  --mount type=bind,source=${HOST_WORKSPACE},target=/home/coder/ \
  -e NVIDIA_DISABLE_REQUIRE=true \
  --gpus device=${NVIDIA_VISIBLE_DEVICES} \
  rapidsai/citestwheel:cuda12.8.0-rockylinux8-py${py_version} \
  bash -c '\
    source /home/coder/cccl/ci/build_common.sh && \
    print_environment_details && \
    fail_if_no_gpu && \
    source /home/coder/cccl/ci/test_python_common.sh && \
    list_environment && \
    # Install the wheel from the artifact location
    ls -la /home/coder/wheelhouse && \
    ls -la /home/coder/wheelhouse && \
    ls -la /home/coder/wheelhouse/cuda_cccl-*.whl && \
    WHEEL_PATH="$(ls /home/coder/wheelhouse/cuda_cccl-*.whl)" && \
    run_tests_from_wheel "cuda_cccl" "$WHEEL_PATH"'
