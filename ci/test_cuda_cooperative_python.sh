#!/bin/bash

set -euo pipefail

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}
echo "Python version: ${py_version}"
echo "Docker socket: " $(ls /var/run/docker.sock)

# Run tests in the same container as the build
docker run --rm \
  --workdir /home/coder/workspace/cccl/python/cuda_cooperative \
  --mount type=bind,source=${HOST_WORKSPACE},target=/home/coder/workspace \
  --gpus device=${NVIDIA_VISIBLE_DEVICES} \
  rapidsai/citestwheel:cuda12.8.0-rockylinux8-py${py_version} \
  bash -c '\
    source "$(dirname "$0")/build_common.sh" && \
    print_environment_details && \
    fail_if_no_gpu && \
    source "test_python_common.sh" && \
    list_environment && \
    # Install the wheel from the artifact location \
    WHEEL_PATH="$(ls /wheelhouse/cuda_cooperative-*.whl)" && \
    run_tests_from_wheel "cuda_cooperative" "$WHEEL_PATH"'
