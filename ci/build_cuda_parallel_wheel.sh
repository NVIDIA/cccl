#!/bin/bash
set -euo pipefail

# Install required tools
dnf -y install unzip
dnf -y install gcc-toolset-13-gcc gcc-toolset-13-gcc-c++

# Enable GCC toolset
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
RAWCPWHL=$(ls cuda_parallel-*.whl | head -n1)

# Collect excluded shared libs from unpacked wheel
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

# Repair wheel
python -m pip install patchelf auditwheel
python -m auditwheel repair $EXCLUDES "$RAWCPWHL"

# Move wheels to output directory
mv cuda_cccl-*.whl /workspace/cccl/wheelhouse/
mv wheelhouse/cuda_parallel-*.whl /workspace/cccl/wheelhouse/
