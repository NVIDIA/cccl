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

# cuda-compute must be built in a container that can produce manylinux wheels,
# and has the CUDA toolkit installed. We use the rapidsai/ci-wheel image for this.
# We build separate wheels using separate containers for each CUDA version,
# then merge them into a single wheel.

readonly cuda12_version=12.9.1
readonly cuda13_version=13.1.1
readonly devcontainer_version=26.04
readonly devcontainer_distro=rockylinux8
# Use a baseline Python tag for the rapidsai ci-wheel image. The requested
# py_version is installed inside the container by setup_python_env (uv).
# Pinning the image tag avoids relying on a per-py_version image being
# published (e.g. py3.14 images may not yet exist).
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

rm -rf wheelhouse wheelhouse_merged wheelhouse_final
mkdir -p wheelhouse

# Shared caches across the cu12 + cu13 wheel builds. Both jobs compile an
# identical LLVM/clang tree (LLVM has no CUDA dep), so a shared ccache cuts
# the second build's LLVM phase from ~10 min to under 2 min; a shared CPM
# source cache skips the second LLVM git clone entirely.
#
# The `mkdir`s run inside the (dev)container where only the container-side
# paths are visible. The docker bind-mount uses the host-side paths
# (${HOST_WORKSPACE}) since the inner docker daemon is the host's.
mkdir -p ./.ccache ./.cpm-cache
host_ccache_dir="${HOST_WORKSPACE:?}/.ccache"
host_cpm_cache_dir="${HOST_WORKSPACE:?}/.cpm-cache"

for ctk in 12 13; do
  image="cuda${ctk}_image"
  image="${!image}"
  build_cuda_cccl_metapackage=0
  if [[ "$ctk" == 12 ]]; then
    build_cuda_cccl_metapackage=1
  fi
  echo "::group::⚒️ Building CUDA $ctk wheel on $image"
  (
    set -x
    docker pull "$image"
    docker run --rm -i \
        --workdir /workspace/python/cuda_compute \
        --mount "type=bind,source=${HOST_WORKSPACE:?},target=/workspace/" \
        --mount "type=bind,source=${host_ccache_dir},target=/root/.ccache" \
        --mount "type=bind,source=${host_cpm_cache_dir},target=/root/.cpm-cache" \
        "${action_mounts[@]}" \
        --env "py_version=${py_version}" \
        --env "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}" \
        --env "GITHUB_RUN_ID=${GITHUB_RUN_ID:-}" \
        --env "JOB_ID=${JOB_ID:-}" \
        --env "CCCL_PYTHON_USE_V2=${CCCL_PYTHON_USE_V2:-}" \
        --env "CCCL_C_PARALLEL_SANITIZE_THREAD=${CCCL_C_PARALLEL_SANITIZE_THREAD:-}" \
        --env "BUILD_CUDA_CCCL_METAPACKAGE=${build_cuda_cccl_metapackage}" \
        --env "CCACHE_DIR=/root/.ccache" \
        --env "CPM_SOURCE_CACHE=/root/.cpm-cache" \
        "$image" \
        /workspace/ci/build_cuda_cccl_wheel.sh
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
python python/cuda_compute/merge_cuda_wheels.py "$cu12_wheel" "$cu13_wheel" --output-dir wheelhouse_merged

# A ThreadSanitizer wheel links libtsan; keep it external (excluded) so it is
# NOT bundled -- the TSan test lane LD_PRELOADs the runner's matching libtsan
# instead. Harmless for normal builds (the .so has no libtsan dependency).
tsan_exclude=()
if [[ "${CCCL_C_PARALLEL_SANITIZE_THREAD:-}" =~ ^(1|true|TRUE|on|ON)$ ]]; then
  tsan_exclude=(--exclude 'libtsan.so.2')
fi

# Install auditwheel and repair the merged wheel
python -m pip install patchelf auditwheel twine
for wheel in wheelhouse_merged/cuda_compute-*.whl; do
    echo "Repairing merged wheel: $wheel"
    python -m auditwheel repair \
        --exclude 'libnvrtc.so.12' \
        --exclude 'libnvrtc.so.13' \
        --exclude 'libnvJitLink.so.12' \
        --exclude 'libnvJitLink.so.13' \
        --exclude 'libcudart.so.12' \
        --exclude 'libcudart.so.13' \
        --exclude 'libcuda.so.1' \
        "${tsan_exclude[@]}" \
        "$wheel" \
        --wheel-dir wheelhouse_final
done

# Remove the CUDA-major intermediates while preserving the universal
# cuda-cccl metapackage built by the CUDA 12 producer.
find wheelhouse -maxdepth 1 -name 'cuda_compute-*.cu*.whl' -delete

# Move only the final repaired merged wheel
if ls wheelhouse_final/cuda_compute-*.whl 1> /dev/null 2>&1; then
    mv wheelhouse_final/cuda_compute-*.whl wheelhouse/
    echo "Final merged wheel moved to wheelhouse"
else
    echo "No final repaired wheel found, moving unrepaired merged wheel"
    mv wheelhouse_merged/cuda_compute-*.whl wheelhouse/
fi

# Clean up temporary directories
rm -rf wheelhouse_merged wheelhouse_final

echo "Final wheels in wheelhouse:"
ls -la wheelhouse/
python -m twine check wheelhouse/*.whl

# Catch missing or stale artifacts before any test consumer downloads them.
test "$(find wheelhouse -maxdepth 1 -name 'cuda_compute-*.whl' | wc -l)" -eq 1
test "$(find wheelhouse -maxdepth 1 -name 'cuda_cccl-*-py3-none-any.whl' | wc -l)" -eq 1

# Native JIT templates and generated sources must remain relocatable. Reject
# paths from either the inner manylinux build mount or this outer checkout.
python - "$(pwd -P)" wheelhouse/*.whl <<'PY'
import sys
import zipfile
from pathlib import Path

checkout_roots = {"/workspace", str(Path(sys.argv[1]).resolve())}
needles = {
    variant.encode()
    for root in checkout_roots
    for variant in (root, root.replace("/", "\\"))
    if len(root) > 1
}
violations = []
for wheel_arg in sys.argv[2:]:
    wheel = Path(wheel_arg)
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.infolist():
            payload = archive.read(member)
            if any(needle in payload for needle in needles):
                violations.append(f"{wheel.name}:{member.filename}")
if violations:
    raise SystemExit(
        "wheel payload contains absolute checkout/build paths: "
        + ", ".join(violations)
    )
PY

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name="$(ci/util/workflow/get_wheel_artifact_name.sh)"
  ci/util/artifacts/upload.sh "$wheel_artifact_name" 'wheelhouse/.*'
fi
