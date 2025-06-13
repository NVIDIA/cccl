#!/bin/bash
set -euo pipefail

# Target script for `docker run` command in build_cuda_cccl_python.sh
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

# Figure out the version to use for the package, we need repo history
if $(git rev-parse --is-shallow-repository); then
  git fetch --unshallow
fi
export PACKAGE_VERSION_PREFIX="0.1."
package_version=$(/workspace/ci/generate_version.sh)
echo "Using package version ${package_version}"
# Override the version used by setuptools_scm to the custom version
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUDA_CCCL="${package_version}"

# Build wheels
python -m pip wheel --no-deps /workspace/python/cuda_cccl

# Repair wheel
python -m pip install patchelf auditwheel
python -m auditwheel repair \
    --exclude 'libcuda.so*' \
    --exclude 'libnvrtc.so*' \
    --exclude 'libnvJitLink.so*' \
    cuda_cccl-*.whl

# Move wheel to output directory
mv wheelhouse/cuda_cccl-*.whl /workspace/wheelhouse/
