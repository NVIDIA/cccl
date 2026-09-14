#!/usr/bin/env bash

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
  if ci/util/version_compare.sh "$v1" ge "$v2"; then
    echo "$v1"
  else
    echo "$v2"
  fi
}

# Get the current CCCL info:
readonly cccl_repo="${PWD}"

# Define CCCL_TAG to override the default CCCL SHA. Otherwise the current HEAD
# of the local checkout is used.
echo "CCCL_TAG (override): ${CCCL_TAG-}";
if test -n "${CCCL_TAG-}"; then
    if [[ "${CCCL_RESOLVE_TAG_LOCALLY:-0}" == 1 ]]; then
      cccl_sha="$(git -C "${cccl_repo}" rev-parse "${CCCL_TAG}^{commit}")";
    else
      git -C "${cccl_repo}" fetch origin "${CCCL_TAG}";
      cccl_sha="$(git -C "${cccl_repo}" rev-parse FETCH_HEAD)";
    fi
else
    cccl_sha="$(git -C "${cccl_repo}" rev-parse HEAD)";
fi

cccl_repo_version="$(git -C "${cccl_repo}" describe "${cccl_sha}"| grep -Eo '[0-9]+\.[0-9]+\.[0-9]+')"
readonly cccl_repo_version

# Define CCCL_VERSION to override the version used by rapids-cmake to patch CCCL.
echo "CCCL_VERSION (override): ${CCCL_VERSION-}";
if test -n "${CCCL_VERSION-}"; then
  readonly cccl_rapids_cmake_version="${CCCL_VERSION}"
else
  cccl_rapids_cmake_version="${cccl_repo_version}"
  # shellcheck disable=SC2034
  readonly cccl_rapids_cmake_version
fi

# If the current version is less than 2.8.0, use 2.8.0 for the rapids-cmake version.
# This is to allow rapids-cmake to correctly patch the CCCL install rules on current `main`.
cccl_version=$(version_max "${cccl_repo_version}" "2.8.0")
readonly cccl_version

readonly workdir="${cccl_repo}/build/${CCCL_BUILD_INFIX:-}/matx"
readonly version_file="${workdir}/MatX/cmake/versions.json"
readonly version_override_file="${workdir}/versions-override.json"

log_vars \
  matx_repo matx_branch \
  cccl_repo cccl_sha cccl_repo_version cccl_rapids_cmake_version \
  workdir \
  version_file version_override_file

mkdir -p "${workdir}"
cd "${workdir}"

# Python deps:
pip install numpy

# Clone MatX
if [[ "${CCCL_REUSE_THIRD_PARTY_SOURCE:-0}" != 1 || ! -d MatX/.git ]]; then
  rm -rf MatX
  git clone "${matx_repo}" -b "${matx_branch}"
else
  echo "Reusing MatX checkout for baseline comparison."
fi

cd MatX
echo "MatX HEAD:"
git log -1 --format=short
cd ..

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
cat "$version_override_file"

# Configure and build
rm -rf build

declare -a compile_time_cmake_args=()
if [[ "${CCCL_COMPILE_TIME_BENCH:-0}" == 1 ]]; then
  compile_time_cmake_args=(
    "-DCMAKE_CUDA_FLAGS=--fdevice-time-trace=-"
    "-DCMAKE_CUDA_COMPILER_LAUNCHER="
    "-DCMAKE_CXX_COMPILER_LAUNCHER="
  )
fi

SCCACHE_NO_DIST_COMPILE=1 cmake \
  -B build -S MatX -G Ninja \
  "-DCMAKE_CUDA_ARCHITECTURES=75;120" \
  "-DRAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE=${version_override_file}" \
  -DMATX_BUILD_TESTS=ON \
  -DMATX_BUILD_EXAMPLES=ON \
  -DMATX_BUILD_BENCHMARKS=ON \
  -DMATX_EN_CUTENSOR=ON \
  "${compile_time_cmake_args[@]}"

# Disabled because `cmake --build -j ""` is invalid, but so is
# `cmake --build -j8`. CMake expects a space between `-j` and
# the numeric argument, or no argument at all.
# shellcheck disable=SC2086
time cmake --build build -j ${PARALLEL_LEVEL:-}
