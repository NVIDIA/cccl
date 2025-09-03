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
  # Note that these mounts use the runner (not the devcontainer) filesystem for
  # source directories because of docker-out-of-docker quirks.
  # The workflow-job GH actions make sure that they exist before running any
  # scripts.
  action_mounts=$(cat <<EOF
    --mount type=bind,source=${ARTIFACT_ARCHIVES},target=${ARTIFACT_ARCHIVES} \
    --mount type=bind,source=${ARTIFACT_UPLOAD_STAGE},target=${ARTIFACT_UPLOAD_STAGE}
EOF
)

else
  # If not running in GitHub Actions, we don't need to set up artifact mounts.
  action_mounts=""
fi

# cuda_cccl must be built in a container that can produce manylinux wheels,
# and has the CUDA toolkit installed. We use the rapidsai/ci-wheel image for this.
# We build separate wheels using separate containers for each CUDA version,
# then merge them into a single wheel.

mkdir -p wheelhouse

echo "Building CUDA 12 wheel..."
(
  set -x
  docker run --rm -i \
      --workdir /workspace/python/cuda_cccl \
      --mount type=bind,source=${HOST_WORKSPACE},target=/workspace/ \
      ${action_mounts} \
      --env py_version=${py_version} \
      --env GITHUB_ACTIONS=${GITHUB_ACTIONS:-} \
      --env GITHUB_RUN_ID=${GITHUB_RUN_ID:-} \
      --env JOB_ID=${JOB_ID:-} \
      rapidsai/ci-wheel:25.10-cuda12.9.1-rockylinux8-py${py_version} \
      /workspace/ci/build_cuda_cccl_wheel.sh
)

echo "Building CUDA 13 wheel..."
(
  set -x
  docker run --rm -i \
      --workdir /workspace/python/cuda_cccl \
      --mount type=bind,source=${HOST_WORKSPACE},target=/workspace/ \
      ${action_mounts} \
      --env py_version=${py_version} \
      --env GITHUB_ACTIONS=${GITHUB_ACTIONS:-} \
      --env GITHUB_RUN_ID=${GITHUB_RUN_ID:-} \
      --env JOB_ID=${JOB_ID:-} \
      rapidsai/ci-wheel:25.10-cuda13.0.0-rockylinux8-py${py_version} \
      /workspace/ci/build_cuda_cccl_wheel.sh
)

echo "Merging CUDA wheels..."

# Needed for unpacking and repacking wheels.
python -m pip install wheel

# Find the built wheels
cu12_wheel=$(find wheelhouse -name "*cu12*.whl" | head -1)
cu13_wheel=$(find wheelhouse -name "*cu13*.whl" | head -1)

if [[ -z "$cu12_wheel" ]]; then
  echo "Error: CUDA 12 wheel not found in wheelhouse/"
  ls -la wheelhouse/
  exit 1
fi

if [[ -z "$cu13_wheel" ]]; then
  echo "Error: CUDA 13 wheel not found in wheelhouse/"
  ls -la wheelhouse/
  exit 1
fi

echo "Found CUDA 12 wheel: $cu12_wheel"
echo "Found CUDA 13 wheel: $cu13_wheel"

# Merge the wheels
python python/cuda_cccl/merge_cuda_wheels.py "$cu12_wheel" "$cu13_wheel" --output-dir wheelhouse_merged

# Install auditwheel and repair the merged wheel
python -m pip install patchelf auditwheel
for wheel in wheelhouse_merged/cuda_cccl-*.whl; do
    echo "Repairing merged wheel: $wheel"
    python -m auditwheel repair \
        --exclude 'libnvrtc.so.12' \
        --exclude 'libnvrtc.so.13' \
        --exclude 'libnvJitLink.so.12' \
        --exclude 'libnvJitLink.so.13' \
        --exclude 'libcuda.so.1' \
        "$wheel" \
        --wheel-dir wheelhouse_final
done

# Clean up intermediate files and move only the final merged wheel to wheelhouse
rm -rf wheelhouse/*  # Clean existing wheelhouse
mkdir -p wheelhouse

# Move only the final repaired merged wheel
if ls wheelhouse_final/cuda_cccl-*.whl 1> /dev/null 2>&1; then
    mv wheelhouse_final/cuda_cccl-*.whl wheelhouse/
    echo "Final merged wheel moved to wheelhouse"
else
    echo "No final repaired wheel found, moving unrepaired merged wheel"
    mv wheelhouse_merged/cuda_cccl-*.whl wheelhouse/
fi

# Clean up temporary directories
rm -rf wheelhouse_merged wheelhouse_final

echo "Final wheels in wheelhouse:"
ls -la wheelhouse/

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name="$(ci/util/workflow/get_wheel_artifact_name.sh)"
  ci/util/artifacts/upload.sh $wheel_artifact_name 'wheelhouse/.*'
fi
