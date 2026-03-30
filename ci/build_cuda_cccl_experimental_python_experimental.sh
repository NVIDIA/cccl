#!/bin/bash
set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage="Usage: $0 -py-version <python_version> [additional options...]"

source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"

# Check if py_version was provided (this script requires it)
require_py_version "$usage" || exit 1

echo "Docker socket: " $(ls /var/run/docker.sock)

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  # Prepare mount points etc for getting artifacts in/out of the container.
  source "$ci_dir/util/artifacts/common.sh"
  action_mounts=$(cat <<EOF
    --mount type=bind,source=${ARTIFACT_ARCHIVES},target=${ARTIFACT_ARCHIVES} \
    --mount type=bind,source=${ARTIFACT_UPLOAD_STAGE},target=${ARTIFACT_UPLOAD_STAGE}
EOF
)

else
  action_mounts=""
fi

readonly cuda12_version=12.9.1
readonly cuda13_version=13.0.2
readonly devcontainer_version=26.02
readonly devcontainer_distro=rockylinux8

if [[ "$(uname -m)" == "aarch64" ]]; then
  readonly cuda12_image=rapidsai/ci-wheel:${devcontainer_version}-cuda${cuda12_version}-${devcontainer_distro}-py${py_version}-arm64
  readonly cuda13_image=rapidsai/ci-wheel:${devcontainer_version}-cuda${cuda13_version}-${devcontainer_distro}-py${py_version}-arm64
else
  readonly cuda12_image=rapidsai/ci-wheel:${devcontainer_version}-cuda${cuda12_version}-${devcontainer_distro}-py${py_version}
  readonly cuda13_image=rapidsai/ci-wheel:${devcontainer_version}-cuda${cuda13_version}-${devcontainer_distro}-py${py_version}
fi

mkdir -p wheelhouse_experimental

for ctk in 12 13; do
  image=$(eval echo \$cuda${ctk}_image)
  echo "::group::⚒️ Building CUDA ${ctk} experimental wheel on ${image}"
  (
    set -x
    docker pull $image
    docker run --rm -i \
        --workdir /workspace/python/cuda_cccl_experimental \
        --mount type=bind,source=${HOST_WORKSPACE},target=/workspace/ \
        ${action_mounts} \
        --env py_version=${py_version} \
        --env GITHUB_ACTIONS=${GITHUB_ACTIONS:-} \
        --env GITHUB_RUN_ID=${GITHUB_RUN_ID:-} \
        --env JOB_ID=${JOB_ID:-} \
        $image \
        /workspace/ci/build_cuda_cccl_experimental_wheel.sh
    # Prevent GHA runners from exhausting available storage with leftover images:
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
      docker rmi -f $image
    fi
  )
  echo "::endgroup::"
done

echo "Merging CUDA experimental wheels..."

# Needed for unpacking and repacking wheels.
python -m pip install wheel

# Find the built wheels
cu12_wheel=$(find wheelhouse_experimental -name "*cu12*.whl" | head -1)
cu13_wheel=$(find wheelhouse_experimental -name "*cu13*.whl" | head -1)

if [[ -z "$cu12_wheel" ]]; then
  echo "Error: CUDA 12 experimental wheel not found in wheelhouse_experimental/"
  ls -la wheelhouse_experimental/
  exit 1
fi

if [[ -z "$cu13_wheel" ]]; then
  echo "Error: CUDA 13 experimental wheel not found in wheelhouse_experimental/"
  ls -la wheelhouse_experimental/
  exit 1
fi

echo "Found CUDA 12 wheel: $cu12_wheel"
echo "Found CUDA 13 wheel: $cu13_wheel"

# Merge the wheels
python python/cuda_cccl_experimental/merge_cuda_wheels.py "$cu12_wheel" "$cu13_wheel" --output-dir wheelhouse_experimental_merged

# Install auditwheel and repair the merged wheel
python -m pip install patchelf auditwheel
for wheel in wheelhouse_experimental_merged/cuda_cccl_experimental-*.whl; do
    echo "Repairing merged wheel: $wheel"
    python -m auditwheel repair \
        --exclude 'libnvrtc.so.12' \
        --exclude 'libnvrtc.so.13' \
        --exclude 'libnvJitLink.so.12' \
        --exclude 'libnvJitLink.so.13' \
        --exclude 'libcuda.so.1' \
        "$wheel" \
        --wheel-dir wheelhouse_experimental_final
done

# Clean up intermediate files and move only the final merged wheel
rm -rf wheelhouse_experimental/*
mkdir -p wheelhouse_experimental

if ls wheelhouse_experimental_final/cuda_cccl_experimental-*.whl 1> /dev/null 2>&1; then
    mv wheelhouse_experimental_final/cuda_cccl_experimental-*.whl wheelhouse_experimental/
    echo "Final merged experimental wheel moved to wheelhouse_experimental"
else
    echo "No final repaired wheel found, moving unrepaired merged wheel"
    mv wheelhouse_experimental_merged/cuda_cccl_experimental-*.whl wheelhouse_experimental/
fi

# Clean up temporary directories
rm -rf wheelhouse_experimental_merged wheelhouse_experimental_final

echo "Final experimental wheels in wheelhouse_experimental:"
ls -la wheelhouse_experimental/

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name="$(ci/util/workflow/get_wheel_artifact_name.sh)_experimental"
  ci/util/artifacts/upload.sh $wheel_artifact_name 'wheelhouse_experimental/.*'
fi
