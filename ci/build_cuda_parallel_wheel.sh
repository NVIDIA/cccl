#!/bin/bash
set -euo pipefail

source /workspace/ci/pyenv_helper.sh
setup_python_env "${py_version}"
echo "Done setting up python env"
python -m pip wheel --no-deps /workspace/python/cuda_cccl
dnf -y install gcc-toolset-13-gcc gcc-toolset-13-gcc-c++
echo -e "#!/bin/bash\nsource /opt/rh/gcc-toolset-13/enable" > /etc/profile.d/enable_devtools.sh
source /etc/profile.d/enable_devtools.sh
which python
python --version
python -m pip wheel --no-deps .
python -m pip install auditwheel
python -m auditwheel repair cuda_parallel-*.whl \
    --exclude libcuda.so.1 \
    --exclude nvrtc.so.12 \
    --exclude nvJitLink.so.12
mv cuda_cccl-*.whl /workspace/cccl/wheelhouse/
mv wheelhouse/cuda_parallel-*.whl /workspace/cccl/wheelhouse/
