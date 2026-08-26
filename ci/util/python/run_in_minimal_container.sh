#!/usr/bin/env bash
# Run a CI script inside a deliberately minimal sibling container.
#
# cuda.compute is supposed to work with nothing installed beyond its declared
# pip dependencies -- no host compiler, no system CUDA toolkit. The CCCL
# devcontainer has all of those, so a test running there cannot tell the
# difference between "we depend only on our wheels" and "we happened to find
# gcc and /usr/local/cuda lying around". This script runs the payload in an
# image that has nothing but Python, so that difference fails loudly.
#
# The job itself still runs in the devcontainer; this launches a sibling
# container through the host's docker daemon, the same docker-outside-of-docker
# arrangement ci/build_cuda_cccl_python.sh uses. Everything the CI harness needs
# (gh to fetch the wheel artifact, jq/tar to stage result artifacts) stays in the
# devcontainer. Only the test payload runs inside the minimal image.
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
# The host daemon resolves the bind-mount source, so it must be a host path:
# HOST_WORKSPACE is what the devcontainer sets for exactly this purpose.
host_workspace="${HOST_WORKSPACE:-$(pwd)}"
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

# Every lane using this path is single-GPU. A multi-GPU lane would need the same
# handling workflow-run-job-linux gives it (--gpus all rather than a device id).
# "void" is the nvidia-container-toolkit spelling of "no GPU", so it names no
# device to hand on.
declare -a gpu_request=()
if [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" && "${NVIDIA_VISIBLE_DEVICES}" != "void" ]]; then
  gpu_request+=(--gpus "device=${NVIDIA_VISIBLE_DEVICES}")
elif nvidia-smi -L &> /dev/null; then
  # GPUs are reachable here, but nothing said which one to pass on. Every lane
  # using this path needs one, and running without would fail confusingly deep
  # inside pytest -- so in CI that is an error. Locally it is common enough
  # (developer devcontainers often leave the variable unset) to be a warning.
  readonly gpu_msg="GPUs are present but NVIDIA_VISIBLE_DEVICES names none (\"${NVIDIA_VISIBLE_DEVICES:-unset}\")"
  if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
    echo "ERROR: ${gpu_msg}." >&2
    exit 1
  fi
  echo "WARNING: ${gpu_msg}; the payload will run without one." >&2
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
