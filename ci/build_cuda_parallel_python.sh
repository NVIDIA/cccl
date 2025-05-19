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
  bash -c '\
    source /workspace/ci/pyenv_helper.sh && \
    setup_python_env "${py_version}" && \
    echo "Done setting up python env" && \
    python -m pip wheel --no-deps /workspace/python/cuda_cccl && \
    dnf -y install gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ && \
    echo -e "#!/bin/bash\nsource /opt/rh/gcc-toolset-13/enable" > /etc/profile.d/enable_devtools.sh && \
    source /etc/profile.d/enable_devtools.sh && \
    which python && \
    python -m pip wheel --no-deps . && \
    python -m pip install patchelf auditwheel && \
    python --version && \
    python -m auditwheel repair cuda_parallel-*.whl --exclude libcuda.so.1 && \
    mv cuda_cccl-*.whl /workspace/cccl/wheelhouse && \
    mv wheelhouse/cuda_parallel-*.whl /workspace/cccl/wheelhouse/'
