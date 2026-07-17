#!/usr/bin/env bash
set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage="Usage: $0 -py-version <python_version> [additional options...]"

# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"

# Check if py_version was provided (this script requires it)
require_py_version "$usage" || exit 1

echo "Docker socket: " "$(ls /var/run/docker.sock)"

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  # Prepare mount points etc for getting artifacts in/out of the container.
  # shellcheck source=ci/util/artifacts/common.sh
  source "$ci_dir/util/artifacts/common.sh"
  # Note that these mounts use the runner (not the devcontainer) filesystem for
  # source directories because of docker-out-of-docker quirks.
  # The workflow-job GH actions make sure that they exist before running any
  # scripts.
  action_mounts=(
    --mount "type=bind,source=${ARTIFACT_ARCHIVES},target=${ARTIFACT_ARCHIVES}"
    --mount "type=bind,source=${ARTIFACT_UPLOAD_STAGE},target=${ARTIFACT_UPLOAD_STAGE}"
  )
else
  # If not running in GitHub Actions, we don't need to set up artifact mounts.
  action_mounts=()
fi

# cuda_stf must be built in a container that can produce manylinux wheels, and
# has the CUDA toolkit installed. We use the rapidsai/ci-wheel image for this.
# We build separate wheels using separate containers for each CUDA version,
# then merge them into a single wheel. CUDASTF is Linux-only.

readonly cuda12_version=12.9.1
readonly cuda13_version=13.1.1
readonly devcontainer_version=26.04
readonly devcontainer_distro=rockylinux8
# Use a baseline Python tag for the rapidsai ci-wheel image. The requested
# py_version is installed inside the container by setup_python_env (uv).
readonly devcontainer_python_version=3.10

if [[ "$(uname -m)" == "aarch64" ]]; then
  cuda12_image="rapidsai/ci-wheel:${devcontainer_version}-cuda${cuda12_version}-${devcontainer_distro}-py${devcontainer_python_version}-arm64"
  cuda13_image="rapidsai/ci-wheel:${devcontainer_version}-cuda${cuda13_version}-${devcontainer_distro}-py${devcontainer_python_version}-arm64"
else
  cuda12_image="rapidsai/ci-wheel:${devcontainer_version}-cuda${cuda12_version}-${devcontainer_distro}-py${devcontainer_python_version}"
  cuda13_image="rapidsai/ci-wheel:${devcontainer_version}-cuda${cuda13_version}-${devcontainer_distro}-py${devcontainer_python_version}"
fi
# shellcheck disable=SC2034
readonly cuda12_image
# shellcheck disable=SC2034
readonly cuda13_image

mkdir -p wheelhouse

# Clear stale STF wheels from a previous run so the per-CTK wheel selection
# below is unambiguous. Leave any co-located wheels (e.g. a cuda_cccl wheel
# staged by a combined producer job) untouched.
rm -f wheelhouse/cuda_stf-*.whl

# Shared caches across the cu12 + cu13 wheel builds. Both jobs compile an
# identical LLVM/clang tree (LLVM has no CUDA dep), so a shared ccache cuts
# the second build's LLVM phase substantially; a shared CPM source cache skips
# the second LLVM git clone entirely.
mkdir -p ./.ccache ./.cpm-cache
host_ccache_dir="${HOST_WORKSPACE:?}/.ccache"
host_cpm_cache_dir="${HOST_WORKSPACE:?}/.cpm-cache"

for ctk in 12 13; do
  image="cuda${ctk}_image"
  image="${!image}"
  echo "::group::⚒️ Building CUDA $ctk cuda-stf wheel on $image"
  (
    set -x
    docker pull "$image"
    docker run --rm -i \
        --workdir /workspace/python/cuda_stf \
        --mount "type=bind,source=${HOST_WORKSPACE:?},target=/workspace/" \
        --mount "type=bind,source=${host_ccache_dir},target=/root/.ccache" \
        --mount "type=bind,source=${host_cpm_cache_dir},target=/root/.cpm-cache" \
        "${action_mounts[@]}" \
        --env "py_version=${py_version}" \
        --env "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}" \
        --env "GITHUB_RUN_ID=${GITHUB_RUN_ID:-}" \
        --env "JOB_ID=${JOB_ID:-}" \
        --env "CCACHE_DIR=/root/.ccache" \
        --env "CPM_SOURCE_CACHE=/root/.cpm-cache" \
        "$image" \
        /workspace/ci/build_cuda_stf_wheel.sh
    # Prevent GHA runners from exhausting available storage with leftover images:
    if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
      docker rmi -f "$image"
    fi
  )
  echo "::endgroup::"
done

echo "Merging CUDA wheels..."

# Set up a Python environment for the merge/repair steps.
source "$ci_dir/pyenv_helper.sh"
setup_python_env "${py_version}"

# Needed for unpacking and repacking wheels.
python -m pip install wheel

# Find the built wheels, requiring exactly one match per CUDA version so a
# stale or duplicate wheel cannot be silently merged.
require_single_wheel() {
  local pattern="$1" desc="$2"
  local matches=()
  while IFS= read -r match; do
    matches+=("$match")
  done < <(find wheelhouse -maxdepth 1 -name "$pattern" | sort)
  if [[ ${#matches[@]} -eq 0 ]]; then
    echo "Error: no $desc cuda-stf wheel found in wheelhouse/ (pattern: $pattern)" >&2
    ls -la wheelhouse/ >&2
    exit 1
  fi
  if [[ ${#matches[@]} -gt 1 ]]; then
    echo "Error: expected exactly one $desc cuda-stf wheel, found ${#matches[@]}:" >&2
    printf '  %s\n' "${matches[@]}" >&2
    exit 1
  fi
  printf '%s\n' "${matches[0]}"
}

cu12_wheel="$(require_single_wheel 'cuda_stf-*cu12*.whl' 'CUDA 12')"
cu13_wheel="$(require_single_wheel 'cuda_stf-*cu13*.whl' 'CUDA 13')"

echo "Found CUDA 12 wheel: $cu12_wheel"
echo "Found CUDA 13 wheel: $cu13_wheel"

# Merge the wheels
python python/cuda_stf/merge_cuda_wheels.py "$cu12_wheel" "$cu13_wheel" --output-dir wheelhouse_merged

# Install auditwheel and repair the merged wheel
python -m pip install patchelf auditwheel
for wheel in wheelhouse_merged/cuda_stf-*.whl; do
    echo "Repairing merged wheel: $wheel"
    python -m auditwheel repair \
        --exclude 'libnvrtc.so.12' \
        --exclude 'libnvrtc.so.13' \
        --exclude 'libnvJitLink.so.12' \
        --exclude 'libnvJitLink.so.13' \
        --exclude 'libcudart.so.12' \
        --exclude 'libcudart.so.13' \
        --exclude 'libcuda.so.1' \
        "$wheel" \
        --wheel-dir wheelhouse_final
done

# Drop only the per-CTK STF inputs we just merged; keep any unrelated wheels
# (e.g. a co-located cuda_cccl wheel) intact.
rm -f "$cu12_wheel" "$cu13_wheel"
mkdir -p wheelhouse

# Move only the final repaired merged wheel
if ls wheelhouse_final/cuda_stf-*.whl 1> /dev/null 2>&1; then
    mv wheelhouse_final/cuda_stf-*.whl wheelhouse/
    echo "Final merged wheel moved to wheelhouse"
else
    echo "No final repaired wheel found, moving unrepaired merged wheel"
    mv wheelhouse_merged/cuda_stf-*.whl wheelhouse/
fi

# Clean up temporary directories
rm -rf wheelhouse_merged wheelhouse_final

echo "Final wheels in wheelhouse:"
ls -la wheelhouse/

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  # Upload under a distinct artifact name so it does not clobber the cuda-cccl
  # wheel (both build jobs run in project 'python').
  wheel_artifact_name="$(CCCL_WHEEL_KIND=stf ci/util/workflow/get_wheel_artifact_name.sh)"
  # Upload only the final STF wheel, not any co-located wheels in wheelhouse/.
  ci/util/artifacts/upload.sh "$wheel_artifact_name" 'wheelhouse/cuda_stf-.*\.whl'
fi
