#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../python/cuda_cccl"

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}
echo "Python version: ${py_version}"
echo "Docker socket: " $(ls /var/run/docker.sock)
ls /usr/bin
which docker

# given the py_version build the wheel and output the artifact
# to the artifacts directory
docker run --rm \
  --workdir /home/coder/workspace/cccl/cccl/python/cuda_cccl \
  --mount type=bind,source=${HOST_WORKSPACE},target=/home/coder/workspace \
  rapidsai/ci-wheel:cuda12.8.0-rockylinux8-py${py_version} \
  bash -c '\
    # dnf -y install gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ && \
    # echo -e "#!/bin/bash\nsource /opt/rh/gcc-toolset-13/enable" > /etc/profile.d/enable_devtools.sh && \
    # source /etc/profile.d/enable_devtools.sh && \
    pwd && \
    ls -la / && \
    ls -la /home/ && \
    ls -la /home/coder/workspace && \
    ls -la /home/coder/workspace/cccl && \
    ls -la /home/coder/workspace/cccl/cccl && \
    ls -la /home/coder/workspace/cccl/cccl/python && \
    ls -la /home/coder/workspace/cccl/cccl/python/cuda_cccl && \
    python -m pip wheel .'

# print wheel name:
wheel_name=$(ls python/cuda_cccl/*.whl)
echo "Wheel name: ${wheel_name}"

# copy the wheel to the artifacts directory
cp ${wheel_name} /wheelhouse/
