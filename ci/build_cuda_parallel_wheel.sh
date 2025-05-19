#!/bin/bash
set -euo pipefail

dnf -y install unzip

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
RAWCPWHL=cuda_parallel-*.whl

TMPDIR=$(mktemp -d)
unzip "$RAWCPWHL" -d "$TMPDIR"
EXCLUDES=$(
    find "$TMPDIR" -name '*.so' -exec ldd {} + |
        grep -E 'libcuda\.so|libnvrtc\.so|libnvJitLink\.so' |
        awk '{print $1}' |
        sort -u |
        xargs -n1 echo --exclude
)
echo "Excluding the following libraries in auditwheel repair:"
echo "$EXCLUDES"
rm -rf "$TMPDIR"

python -m pip install patchelf auditwheel
python -m auditwheel repair $EXCLUDES "$RAWCPWHL"
mv cuda_cccl-*.whl /workspace/cccl/wheelhouse/
mv wheelhouse/cuda_parallel-*.whl /workspace/cccl/wheelhouse/
