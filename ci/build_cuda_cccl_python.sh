#!/usr/bin/env bash
set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage="Usage: $0 -py-version <python_version> [additional options...]"

# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"

# Check if py_version was provided (this script requires it)
require_py_version "$usage" || exit 1

# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"

if cccl_python_is_windows; then
  # The matrix's pybuild image supplies CUDA 12. CUDA 13 uses a second native
  # Windows image, sharing the runner's checkout through Docker-out-of-Docker.
  # The image persists its MSVC developer environment at machine scope.
  # Discard this builder's intermediates so a failed/retried invocation cannot
  # merge or publish a wheel produced by an earlier run.
  rm -f wheelhouse/cuda_cccl-*.cu12.whl wheelhouse/cuda_cccl-*.cu13.whl \
    python/cuda_cccl/dist/cuda_cccl-*.whl \
    wheelhouse_merged/cuda_cccl-*.whl wheelhouse_final/cuda_cccl-*.whl
  setup_python_env "${py_version}" .cccl-wheel-merge-venv
  CCCL_WHEEL_CUDA_MAJOR=12 bash --noprofile --norc "$ci_dir/build_cuda_cccl_wheel.sh"

  : "${HOST_WORKSPACE:?required for the nested Windows wheel build}"
  : "${CONTAINER_WORKSPACE:?required for the nested Windows wheel build}"
  host_workspace=$(cygpath -am "$HOST_WORKSPACE")
  container_workspace=$(cygpath -am "$CONTAINER_WORKSPACE")
  cuda13_image="${CCCL_CUDA13_WHEEL_IMAGE:-rapidsai/devcontainers:26.06-cuda13.0-cl14.44-windows2022}"
  cpu_count="${NUMBER_OF_PROCESSORS:?}"
  # Match the existing native wheel builder's headroom for its outer container.
  memory_gb=$(python - <<'PY'
import ctypes
import struct

status = ctypes.create_string_buffer(64)
struct.pack_into("I", status, 0, len(status))
if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
    raise ctypes.WinError()
total_bytes = struct.unpack_from("Q", status, 8)[0]
print(max(2, int(total_bytes / 1024**3 * 0.9)))
PY
  )
  memory_gb=${memory_gb%$'\r'}
  (
    # All Docker paths below are already native. MSYS must not rewrite mount
    # specifications or the inner container's paths using the outer Git root.
    export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
    docker version
    docker run --rm -i \
      --cpu-count "$cpu_count" --memory "${memory_gb}g" \
      --workdir "$container_workspace" \
      --mount "type=bind,source=${host_workspace},target=${container_workspace}" \
      --env "py_version=${py_version}" \
      --env CCCL_WHEEL_CUDA_MAJOR=13 \
      --env "CCCL_PYTHON_USE_V2=${CCCL_PYTHON_USE_V2:-}" \
      "$cuda13_image" \
      'C:/Program Files/Git/bin/bash.exe' --noprofile --norc \
      "${container_workspace}/ci/build_cuda_cccl_wheel.sh"
  )
else
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

  # cuda_cccl must be built in a container that can produce manylinux wheels,
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
    echo "::group::⚒️ Building CUDA $ctk wheel on $image"
    (
      set -x
      docker pull "$image"
      docker run --rm -i \
          --workdir /workspace/python/cuda_cccl \
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
fi

echo "Merging CUDA wheels..."

# Set up a Python environment for the merge/repair steps.
if ! cccl_python_is_windows; then
  setup_python_env "${py_version}"
fi

# Needed for unpacking and repacking wheels.
python -m pip install wheel

# Find the built wheels
if cccl_python_is_windows; then
  cu12_wheels=(wheelhouse/cuda_cccl-*.cu12.whl)
  cu13_wheels=(wheelhouse/cuda_cccl-*.cu13.whl)
  if [[ ${#cu12_wheels[@]} != 1 || ! -f "${cu12_wheels[0]}" ||
        ${#cu13_wheels[@]} != 1 || ! -f "${cu13_wheels[0]}" ]]; then
    echo "Error: expected exactly one newly built wheel per CUDA major" >&2
    exit 1
  fi
  cu12_wheel=${cu12_wheels[0]}
  cu13_wheel=${cu13_wheels[0]}
else
  cu12_wheel=$(find wheelhouse -name "*cu12*.whl" | head -1)
  cu13_wheel=$(find wheelhouse -name "*cu13*.whl" | head -1)
fi

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

if cccl_python_is_windows; then
  # Analyze DLLs already inside the wheel as well as its Python extensions,
  # and preserve the shared cuda namespace when installing the loader patch.
  python -m pip install 'delvewheel>=1.13.1'
  system32=$(cygpath -am "${SystemRoot:-${SYSTEMROOT:?}}/System32")
  test -f "$(cygpath -u "$system32")/msvcp140.dll"
  for wheel in wheelhouse_merged/cuda_cccl-*.whl; do
    python -m delvewheel repair "$wheel" --wheel-dir wheelhouse_final \
      --analyze-existing --namespace-pkg cuda \
      --exclude 'cccl.c.parallel*.dll;libnvcc.dll;nvrtc64_*.dll;nvrtc-builtins64_*.dll;nvJitLink_*.dll;nvfatbin*.dll;cudart64_*.dll;nvcuda.dll;dbghelp.dll' \
      --add-path "$system32"
  done
else
  # A ThreadSanitizer wheel links libtsan; keep it external (excluded) so it is
  # NOT bundled -- the TSan test lane LD_PRELOADs the runner's matching libtsan
  # instead. Harmless for normal builds (the .so has no libtsan dependency).
  tsan_exclude=()
  if [[ "${CCCL_C_PARALLEL_SANITIZE_THREAD:-}" =~ ^(1|true|TRUE|on|ON)$ ]]; then
    tsan_exclude=(--exclude 'libtsan.so.2')
  fi

  # Install auditwheel and repair the merged wheel
  python -m pip install patchelf auditwheel
  for wheel in wheelhouse_merged/cuda_cccl-*.whl; do
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
fi

# Clean up intermediate files and move only the final merged wheel to wheelhouse
rm -rf wheelhouse/*  # Clean existing wheelhouse
mkdir -p wheelhouse

# Move only the final repaired merged wheel
if ls wheelhouse_final/cuda_cccl-*.whl 1> /dev/null 2>&1; then
    mv wheelhouse_final/cuda_cccl-*.whl wheelhouse/
    echo "Final merged wheel moved to wheelhouse"
elif cccl_python_is_windows; then
    echo "Error: delvewheel did not produce a repaired wheel" >&2
    exit 1
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
  ci/util/artifacts/upload.sh "$wheel_artifact_name" 'wheelhouse/.*'
fi
