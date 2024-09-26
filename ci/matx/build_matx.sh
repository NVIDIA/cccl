#!/bin/bash

set -euo pipefail

readonly matx_repo=https://github.com/NVIDIA/MatX.git
readonly matx_branch=main

# Ensure the script is being executed in the root cccl directory:
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../..";

log_vars() {
  for var in "$@"; do
    echo "${var}=${!var}"
  done
}

# Take two version strings and return the greater of the two:
version_max() {
  local v1="${1}"
  local v2="${2}"
  if [[ "$(printf "%s\n" "${v1}" "${v2}" | sort -V | head -n1)" == "${v1}" ]]; then
    echo "${v2}"
  else
    echo "${v1}"
  fi
}

# Get the current CCCL info:
readonly cccl_repo="${PWD}"
readonly cccl_sha="$(git rev-parse HEAD)"
readonly cccl_repo_version="$(git describe | grep -Eo '[0-9]+\.[0-9]+\.[0-9]+')"

# If the current version is less than 2.8.0, use 2.8.0 for the rapids-cmake version.
# This is to allow rapids-cmake to correctly patch the CCCL install rules on current `main`.
readonly cccl_version=$(version_max "${cccl_repo_version}" "2.8.0")

readonly workdir="${cccl_repo}/build/${CCCL_BUILD_INFIX:-}/matx"
readonly version_file="${workdir}/MatX/cmake/versions.json"
readonly version_override_file="${workdir}/versions-override.json"

log_vars \
  matx_repo matx_branch \
  cccl_repo cccl_sha cccl_repo_version cccl_version \
  workdir \
  version_file version_override_file

mkdir -p "${workdir}"
cd "${workdir}"

# Python deps:
pip install numpy

# Clone MatX
rm -rf MatX
git clone ${matx_repo} -b ${matx_branch}

# Write out version override file
jq -r ".packages.CCCL *=
  {
    \"git_url\": \"${cccl_repo}\",
    \"git_tag\": \"${cccl_sha}\",
    \"version\": \"${cccl_version}\",
    \"always_download\": true
  }" \
  "${version_file}" > "${version_override_file}"

echo "Overriding MatX versions.json file:"
cat $version_override_file

# Configure and build
rm -rf build
mkdir build
cd build
cmake -G Ninja ../MatX \
  "-DCMAKE_CUDA_ARCHITECTURES=60;70;80" \
  "-DRAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE=${version_override_file}" \
  -DMATX_BUILD_TESTS=ON \
  -DMATX_BUILD_EXAMPLES=ON \
  -DMATX_BUILD_BENCHMARKS=ON \
  -DMATX_EN_CUTENSOR=ON

cmake --build . -j 8
