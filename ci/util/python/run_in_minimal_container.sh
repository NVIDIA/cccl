#!/usr/bin/env bash
# Re-run a CI script inside a deliberately minimal container.
#
# cuda.compute is supposed to work with nothing installed beyond its declared
# pip dependencies -- no host compiler, no system CUDA toolkit. The CCCL
# devcontainer has all of those, so a test running there cannot tell the
# difference between "we depend only on our wheels" and "we happened to find
# gcc and /usr/local/cuda lying around". This script runs the payload in an
# image that has nothing but Python, so that difference fails loudly.
#
# Everything the CI harness needs (gh to fetch the wheel artifact, jq/tar to
# stage result artifacts) stays OUTSIDE this container, on the runner. Only the
# test payload runs inside.
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

# The workspace is mounted at the devcontainer's workspace path so that scripts
# which hardcode it keep working in both environments.
readonly container_workspace="/home/coder/cccl"
host_workspace="${HOST_WORKSPACE:-${GITHUB_WORKSPACE:-$(pwd)}}"
readonly host_workspace

# The script to run is given as a host path (or a path relative to the repo
# root). Translate an absolute host path into the container's view of the same
# file; relative paths already resolve against the matching workdir.
script="$1"
shift
if [[ "${script}" == /* ]]; then
  script="${container_workspace}/${script#"${host_workspace}"/}"
fi
readonly script

# Every lane that opts into this path is single-GPU. If a multi-GPU Python lane
# is ever added, this needs the same treatment workflow-run-job-linux gives it.
declare -a gpu_request=()
if [[ -n "${NVIDIA_VISIBLE_DEVICES:-}" ]]; then
  gpu_request+=(--gpus "device=${NVIDIA_VISIBLE_DEVICES}")
fi

# Forward only what the payload legitimately needs. In particular CCCL_CUDA_VERSION
# replaces the `nvcc --version` probe that pin_cuda_toolkit uses elsewhere -- there
# is no nvcc in here, by design.
declare -a env_args=(
  # Only so pretty_printing.sh keeps emitting ::group:: markers; nothing on this
  # side of the handoff touches the artifact or workflow helpers.
  --env "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}"
  --env "CCCL_INSIDE_MINIMAL_CONTAINER=1"
  --env "CCCL_CUDA_VERSION=${CCCL_CUDA_VERSION:-${JOB_CUDA:-}}"
  # Load-bearing: test_cuda_compute_python_v2.sh sets this and then execs the v1
  # script, so by the time we hand off, this flag is all that distinguishes them.
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
