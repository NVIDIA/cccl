#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../python/cuda_parallel"

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}
echo "Python version: ${py_version}"
echo "Docker socket: " $(ls /var/run/docker.sock)

# given the py_version build the wheel and output the artifact
# to the artifacts directory
docker run --rm \
  --workdir /workspace/cccl/python/cuda_parallel \
  --mount type=bind,source=${HOST_WORKSPACE},target=/workspace/ \
  rapidsai/ci-wheel:cuda12.8.0-rockylinux8-py${py_version} \
  bash -c '\
    python -m pip wheel --no-deps ../cuda_cccl && \
    dnf -y install gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ && \
    echo -e "#!/bin/bash\nsource /opt/rh/gcc-toolset-13/enable" > /etc/profile.d/enable_devtools.sh && \
    source /etc/profile.d/enable_devtools.sh && \
    python -m pip wheel --no-deps . && \
    auditwheel repair cuda_parallel-*.whl --exclude libcuda.so.1 && \
    mv cuda_cccl-*.whl /workspace/wheelhouse/ && \
    mv wheelhouse/cuda_parallel-*.whl /workspace/wheelhouse/'
