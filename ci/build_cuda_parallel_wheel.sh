#!/bin/bash
set -euo pipefail

# Target script for `docker run` command in build_cuda_parallel_python.sh
# The /workspace pathnames are hard-wired here.

# Enable GCC toolset
dnf -y install gcc-toolset-13-gcc gcc-toolset-13-gcc-c++
echo -e "#!/bin/bash\nsource /opt/rh/gcc-toolset-13/enable" >/etc/profile.d/enable_devtools.sh
source /etc/profile.d/enable_devtools.sh

# Set up Python environment
source /workspace/ci/pyenv_helper.sh
setup_python_env "${py_version}"
which python
python --version
echo "Done setting up python env"

# Build wheels
python -m pip wheel --no-deps /workspace/python/cuda_cccl
python -m pip wheel --no-deps .

# Repair wheel
python -m pip install patchelf auditwheel
python -m auditwheel repair \
    --exclude 'libcuda.so*' \
    --exclude 'libnvrtc.so*' \
    --exclude 'libnvJitLink.so*' \
    cuda_parallel-*.whl

# Move wheels to output directory
mv cuda_cccl-*.whl /workspace/wheelhouse/
mv wheelhouse/cuda_parallel-*.whl /workspace/wheelhouse/
