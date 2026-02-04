#!/bin/bash

set -euo pipefail

readonly pytorch_repo=https://github.com/pytorch/pytorch.git
readonly pytorch_branch=main

# Ensure the script is being executed in the root cccl directory:
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../..";
readonly cccl_repo="${PWD}"

log_vars() {
  for var in "$@"; do
    echo "${var}=${!var}"
  done
}

# Define CCCL_TAG to override the default CCCL SHA. Otherwise the current HEAD of the local checkout is used.
echo "CCCL_TAG (override): ${CCCL_TAG-}";
if test -n "${CCCL_TAG-}"; then
    # If CCCL_TAG is defined, fetch it to the local checkout
    git -C "${cccl_repo}" fetch origin "${CCCL_TAG}";
    cccl_sha="$(git -C "${cccl_repo}" rev-parse FETCH_HEAD)";
else
    cccl_sha="$(git -C "${cccl_repo}" rev-parse HEAD)";
fi

readonly workdir="${cccl_repo}/build/${CCCL_BUILD_INFIX:-}/pytorch"

log_vars \
  pytorch_repo pytorch_branch \
  cccl_repo cccl_sha \
  workdir

mkdir -p "${workdir}"
cd "${workdir}"
echo "Working in ${workdir}"

echo "::group::Cloning CCCL..."
rm -rf cccl
git clone "${cccl_repo}"
git -C cccl checkout "${cccl_sha}"
echo "CCCL HEAD:"
git -C cccl log -1 --format=short
echo "::endgroup::"

# Setup a CUDA environment with the requested CCCL.
# Use a local directory to avoid modifying the actual CUDA install:
echo "::group::Setting up clone of CUDA environment with custom CCCL..."
(
  set -x
  rm -rf ./cuda
  cp -Hr /usr/local/cuda ./cuda
  rm -rf ./cuda/include/cccl/*
  cccl/ci/install_cccl.sh ./cccl-install > /dev/null
  cp -r ./cccl-install/include/* ./cuda/include/cccl
)
export PATH="$PWD/cuda/bin:$PATH"
export CUDA_HOME="$PWD/cuda"
export CUDA_PATH="$PWD/cuda"
which nvcc
nvcc --version
echo "::endgroup::"

echo "::group::Cloning PyTorch..."
rm -rf pytorch
git clone ${pytorch_repo} -b ${pytorch_branch} --recursive --depth 1
echo "PyTorch HEAD:"
git -C pytorch log -1 --format=short
echo "::endgroup::"

echo "::group::Installing PyTorch build dependencies..."
pytorch_root="$PWD/pytorch"
export PYTHONPATH="${pytorch_root}:${pytorch_root}/tools:${PYTHONPATH:-}"
pip install -r "${pytorch_root}/requirements-build.txt"
echo "::endgroup::"

echo "::group::Configuring PyTorch..."
rm -rf build
mkdir build
declare -a cmake_args=(
  "-DUSE_NCCL=OFF"
  # Need to define this explicitly, torch's FindCUDA logic adds ancient arches if left undefined:
  "-DTORCH_CUDA_ARCH_LIST=7.5;8.0;9.0;10.0;12.0"
)
cmake -S ./pytorch -B ./build -G Ninja "${cmake_args[@]}"
echo "::endgroup::"

# Verify that the configured build is using the custom CUDA dir for CTK and nvcc:
if ! grep -q "CUDA_TOOLKIT_ROOT_DIR:PATH=$PWD/cuda" ./build/CMakeCache.txt; then
    echo "Error: CUDA_TOOLKIT_ROOT_DIR does not point to the custom CUDA";
    exit 1;
fi
if ! grep -q "CUDA_NVCC_EXECUTABLE:FILEPATH=$PWD/cuda/bin/nvcc" ./build/CMakeCache.txt; then
    echo "Error: CUDA_NVCC_EXECUTABLE does not point to the custom CUDA";
    exit 1;
fi

# This builds a bunch of unnecessary targets. Leaving here to use as a fallback if the
# ninja target extraction below starts failing:
# echo "::group::Building torch_cuda target..."
# cmake --build ./build/ --target torch_cuda
# echo "::endgroup::"

# This cuts the number of built targets roughly in half:
echo "::group::Extracting cuda targets from build.ninja..."
# Query ninja for all object files built from CUDA source files
# that are part of the torch_cuda library:
ninja -C ./build -t query lib/libtorch_cuda.so |
  grep -E "torch_cuda\\.dir/.*\\.cu\\.o$" |
  sort | uniq | tee build/cuda_targets.txt
# At the time this script was written, there were 311 cuda targets.
# Check that there are at least 100 detected targets, otherwise fail.
num_targets=$(wc -l < build/cuda_targets.txt)
if test "$num_targets" -lt 100; then
    echo "Error: extracted cuda targets count is less than 100! ($num_targets)";
    echo "This likely indicates a failure to extract the targets from ninja.";
    exit 1;
fi
echo "::endgroup::"

echo "::group::Building $num_targets pytorch CUDA targets with custom CCCL..."
ninja -C ./build $(xargs -a build/cuda_targets.txt)
echo "::endgroup::"

echo "PyTorch CUDA targets built successfully with custom CCCL."
