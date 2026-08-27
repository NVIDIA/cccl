#!/usr/bin/env bash
# Run a CI script inside a deliberately minimal sibling container. See "Testing
# Python in a minimal container" in
# docs/infrastructure/ci/references/ci_overview.rst for what this buys and why
# the lanes are split this way.
#
# Note that a matrix `environment:` entry reaches the devcontainer but not this
# container; add it to env_args below if a lane ever needs one.
#
# Usage: run_in_minimal_container.sh <script> [args...]

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <script> [args...]" >&2
  exit 1
fi

# The base image only provides the interpreter used to bootstrap uv; uv then
# installs the exact Python the lane asked for (including free-threaded builds),
# exactly as it does in the devcontainer.
readonly image="${CCCL_MINIMAL_CONTAINER_IMAGE:-python:3.14-slim}"

# Any fixed path works; this matches the devcontainer's for familiarity.
readonly container_workspace="/home/coder/cccl"
# The host daemon resolves the bind-mount source, so it must be a host path.
# HOST_WORKSPACE is what the devcontainer sets for exactly this purpose; running
# straight on the host, the working directory already is one.
if [[ -n "${HOST_WORKSPACE:-}" ]]; then
  host_workspace="${HOST_WORKSPACE}"
elif [[ ! -f /.dockerenv ]]; then
  host_workspace="$(pwd)"
else
  echo "ERROR: HOST_WORKSPACE is not set, so the host daemon would be handed a path only this container can see." >&2
  echo "       Set it, or set CCCL_MINIMAL_CONTAINER=0 to run the payload here." >&2
  exit 1
fi
readonly host_workspace

# Repo-relative, so it resolves the same on both sides of the mount. An absolute
# host path would name a location the container cannot see.
script="$1"
shift
if [[ "${script}" == /* ]]; then
  echo "ERROR: '${script}' must be a path relative to the repo root." >&2
  exit 1
fi
readonly script

# Hand the sibling exactly the GPUs this devcontainer can see, named explicitly:
# `--gpus all` would reach every GPU on the host, including any assigned to a
# different job. NVIDIA_VISIBLE_DEVICES cannot be used to discover them -- the
# devcontainer image pins it to "void" -- so ask the driver.
declare -a gpu_request=()
gpu_uuids="$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null | paste -sd, - || true)"
readonly gpu_uuids
if [[ -n "${gpu_uuids}" ]]; then
  # The inner quotes are load-bearing: docker splits the --gpus value on commas
  # unless the device list is quoted within the argument itself.
  gpu_request+=(--gpus "\"device=${gpu_uuids}\"")
fi

# The whole approach rests on the devcontainer being able to reach the host
# daemon; say so plainly rather than failing inside `docker run`.
if [[ ! -S /var/run/docker.sock ]]; then
  echo "ERROR: /var/run/docker.sock is not available; cannot launch a sibling container." >&2
  exit 1
fi

# Forward only what the payload legitimately needs.
declare -a env_args=(
  # Only so pretty_printing.sh keeps emitting ::group:: markers.
  --env "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}"
  # There is no nvcc in here; pin_cuda_toolkit falls back to this.
  --env "CCCL_CUDA_VERSION=${CCCL_CUDA_VERSION:-}"
  # The v2 entry point sets this and then execs the v1 script, so this flag is
  # the only thing distinguishing a v2 run from a v1 one by the time we get here.
  --env "CCCL_PYTHON_USE_V2=${CCCL_PYTHON_USE_V2:-}"
  # nvidia-container-toolkit only injects libcuda.so.1 when `compute` is
  # requested. The nvidia/cuda images set this for you; a plain image does not.
  --env "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
  # The workspace is bind-mounted; do not litter it with the sibling's bytecode.
  --env "PYTHONDONTWRITEBYTECODE=1"
)

echo "::group::🐍 Running in minimal container: ${image}"
(
  set -x
  docker run --rm -i \
    "${gpu_request[@]}" \
    "${env_args[@]}" \
    --mount "type=bind,source=${host_workspace},target=${container_workspace}" \
    --workdir "${container_workspace}" \
    "${image}" \
    "${script}" "$@"
)
echo "::endgroup::"
