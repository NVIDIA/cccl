#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../python/cuda_cccl"

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}
echo "Python version: ${py_version}"
echo "Docker socket: " $(ls /var/run/docker.sock)

# given the py_version build the wheel and output the artifact
# to the artifacts directory
docker run --rm \
  --workdir /home/coder/workspace/cccl/python/cuda_cccl \
  --mount type=bind,source=${HOST_WORKSPACE},target=/home/coder/workspace \
  rapidsai/ci-wheel:cuda12.8.0-rockylinux8-py${py_version} \
  bash -c '\
    python -m pip wheel --no-deps . && \
    wheel_name=$(ls *.whl) && \
    cp ${wheel_name} /home/coder/workspace/wheelhouse/'
