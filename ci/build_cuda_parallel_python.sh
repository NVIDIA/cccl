#!/bin/bash
set -euo pipefail

# Get the Python version from the command line arguments -py-version=3.10
py_version=${2#*=}
echo "Docker socket: " $(ls /var/run/docker.sock)

# given the py_version build the wheel and output the artifact
# to the artifacts directory
docker run --rm -i \
  --workdir /workspace/cccl/python/cuda_parallel \
  --mount type=bind,source=${HOST_WORKSPACE},target=/workspace/ \
  --env py_version=${py_version} \
  rapidsai/ci-wheel:cuda12.8.0-rockylinux8-py3.10 \
  bash -c '\
    source /workspace/cccl/ci/pyenv_helper.sh && \
    setup_python_env "${py_version}" && \
    echo "Done setting up python env" && \
    python -m pip wheel --no-deps ../cuda_cccl && \
    dnf -y install gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ && \
    echo -e "#!/bin/bash\nsource /opt/rh/gcc-toolset-13/enable" > /etc/profile.d/enable_devtools.sh && \
    source /etc/profile.d/enable_devtools.sh && \
    which python && \
    python -m pip wheel --no-deps . && \
    python -m pip install patchelf auditwheel && \
    python -m auditwheel repair cuda_parallel-*.whl --exclude libcuda.so.1 && \
    mv cuda_cccl-*.whl /workspace/wheelhouse/ && \
    mv wheelhouse/cuda_parallel-*.whl /workspace/wheelhouse/'
