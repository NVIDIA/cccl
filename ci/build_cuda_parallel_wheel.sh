#!/bin/bash
set -euo pipefail

dnf -y install gcc-toolset-13-gcc gcc-toolset-13-gcc-c++
echo -e "#!/bin/bash\nsource /opt/rh/gcc-toolset-13/enable" >/etc/profile.d/enable_devtools.sh
source /etc/profile.d/enable_devtools.sh

source /workspace/ci/pyenv_helper.sh
setup_python_env "${py_version}"
which python
python --version
echo "Done setting up python env"

python -m pip wheel --no-deps /workspace/python/cuda_cccl
python -m pip wheel --no-deps .
python -m pip install patchelf auditwheel
EXCLUDES=$(
    python -m auditwheel show cuda_parallel-*.whl |
        grep -E 'libcuda\.so|libnvrtc\.so|libnvJitLink\.so' |
        awk '{print $1}' |
        sort -u |
        xargs -n1 echo --exclude
)
python -m auditwheel repair $EXCLUDES cuda_parallel-*.whl
mv cuda_cccl-*.whl /workspace/cccl/wheelhouse/
mv wheelhouse/cuda_parallel-*.whl /workspace/cccl/wheelhouse/
