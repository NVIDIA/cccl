#!/bin/bash
set -euo pipefail

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}
echo "Docker socket: " $(ls /var/run/docker.sock)

# cuda_parallel must be built in a container that can produce manylinux wheels,
# and has the CUDA toolkit installed. We use the rapidsai/ci-wheel image for this.
# These images don't come with a new enough version of gcc installed, so that
# must be installed manually.
docker run --rm -i \
    --workdir /workspace/python/cuda_parallel \
    --mount type=bind,source=${HOST_WORKSPACE},target=/workspace/ \
    --env py_version=${py_version} \
    rapidsai/ci-wheel:cuda12.9.0-rockylinux8-py3.10 \
    /workspace/ci/build_cuda_parallel_wheel.sh
