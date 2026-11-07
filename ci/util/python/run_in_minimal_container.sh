#!/usr/bin/env bash
# Run a CI script inside a deliberately minimal sibling container. See "Testing
# Python in a minimal container" in
# docs/infrastructure/ci/references/ci_scripts.rst for what this buys and why
# the lanes are split this way.
#
# Note that a matrix `environment:` entry reaches the devcontainer but not this
# container; add it to env_args below if a lane ever needs one.
#
# Usage: run_in_minimal_container.sh <script> [args...]

set -euo pipefail

windows=false
case "${OSTYPE}" in
  msys* | mingw*) windows=true ;;
  *) ;;
esac

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <script> [args...]" >&2
  exit 1
fi

# uv installs the requested Python, including free-threaded builds. Linux has
# a bootstrap interpreter; Server Core uses its native PowerShell bootstrap.
if $windows; then
  # GPU passthrough needs process isolation and the same kernel as the host.
  image="${CCCL_MINIMAL_CONTAINER_IMAGE:-mcr.microsoft.com/windows/servercore:ltsc2022}"
  container_workspace="${CONTAINER_WORKSPACE:?CONTAINER_WORKSPACE is required on Windows}"
  container_workspace=$(cygpath -am "$container_workspace")
else
  image="${CCCL_MINIMAL_CONTAINER_IMAGE:-python:3.14-slim}"
  container_workspace="/home/coder/cccl"
fi
readonly image container_workspace

# The host daemon resolves the bind-mount source, so it must be a host path.
# HOST_WORKSPACE is what the devcontainer sets for exactly this purpose; running
# straight on the host, the working directory already is one.
if [[ -n "${HOST_WORKSPACE:-}" ]]; then
  host_workspace="${HOST_WORKSPACE}"
elif $windows; then
  echo "ERROR: HOST_WORKSPACE is required for the Windows sibling container's bind mount." >&2
  exit 1
elif [[ ! -f /.dockerenv ]]; then
  host_workspace="$(pwd)"
else
  echo "ERROR: HOST_WORKSPACE is not set, so the host daemon would be handed a path only this container can see." >&2
  echo "       Set it, or set CCCL_MINIMAL_CONTAINER=0 to run the payload here." >&2
  exit 1
fi
if $windows; then
  host_workspace=$(cygpath -am "$host_workspace")
fi
readonly host_workspace

# Repo-relative, so it resolves the same on both sides of the mount. An absolute
# host path would name a location the container cannot see.
script="$1"
shift
if [[ "${script}" == /* || "${script}" == \\* || "${script}" =~ ^[[:alpha:]]: ]]; then
  echo "ERROR: '${script}' must be a path relative to the repo root." >&2
  exit 1
fi
readonly script

# Hand the sibling exactly the GPUs this devcontainer can see, named explicitly:
# `--gpus all` would reach every GPU on the host, including any assigned to a
# different job. NVIDIA_VISIBLE_DEVICES cannot be used to discover them -- the
# devcontainer image pins it to "void" -- so ask the driver.
# Every lane that reaches this helper is a GPU lane, so failing to enumerate is
# fatal rather than silently handing over a GPU-less container -- that surfaces
# much later, as confusing failures inside pytest. nvidia-smi's own diagnostic is
# left on stderr. CCCL_MINIMAL_CONTAINER_NO_GPU=1 opts out, which the headers
# lane uses -- it asserts on files shipped in the wheel and launches no kernel.
declare -a gpu_request=()
if [[ "${CCCL_MINIMAL_CONTAINER_NO_GPU:-}" != "1" ]]; then
  if $windows; then
    # Windows exposes the GPU device class, rather than individual UUIDs.
    gpu_request+=(--device class/5B45201D-F2F2-4F3B-85BB-30FF1F953599)
  else
    if ! gpu_uuids="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | paste -sd, -)" ||
       [[ -z "${gpu_uuids}" ]]; then
      echo "ERROR: could not enumerate GPUs with nvidia-smi, but the payload needs one." >&2
      echo "       Set CCCL_MINIMAL_CONTAINER_NO_GPU=1 to run without a GPU, or" >&2
      echo "       CCCL_MINIMAL_CONTAINER=0 to run the payload in the devcontainer." >&2
      exit 1
    fi
    readonly gpu_uuids
    # The inner quotes keep Docker from splitting the device list on commas.
    gpu_request+=(--gpus "\"device=${gpu_uuids}\"")
  fi
fi

# The whole approach rests on the devcontainer being able to reach the host
# daemon; say so plainly rather than failing inside `docker run`.
if $windows; then
  docker version > /dev/null
elif [[ ! -S /var/run/docker.sock ]]; then
  echo "ERROR: /var/run/docker.sock is not available; cannot launch a sibling container." >&2
  exit 1
fi

# The sibling has no toolkit. Resolve its version before leaving the builder.
cuda_version="${CCCL_CUDA_VERSION:-}"
if [[ -z "$cuda_version" ]] && command -v nvcc > /dev/null; then
  cuda_version=$(nvcc --version | sed -nE 's/.*release ([0-9]+\.[0-9]+).*/\1/p')
fi
if ! [[ "$cuda_version" =~ ^[0-9]+\.[0-9]+$ ]]; then
  echo "ERROR: cannot determine the CUDA version for the minimal container." >&2
  exit 1
fi

declare -a platform_args=()
declare -a payload=("${script}" "$@")
if $windows; then
  # Server Core has no shell runtime. Stage only Git Bash's tools and their
  # data in the shared workspace; never copy Python, MSVC or the CUDA toolkit.
  repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
  runtime_dir=$(mktemp -d "${repo_root}/.cccl-minimal-bash.XXXXXX")
  trap 'rm -rf "$runtime_dir"' EXIT
  mkdir -p "$runtime_dir/usr" "$runtime_dir/mingw64" "$runtime_dir/etc" "$runtime_dir/tmp"
  cp -R /usr/bin "$runtime_dir/usr/"
  cp -R /mingw64/bin "$runtime_dir/mingw64/"
  # Do not copy /etc/fstab or shell profiles: their original-image paths can
  # override the relocated runtime's mount table or add tools to PATH.
  for data_dir in /etc/ssl /mingw64/etc /mingw64/ssl; do
    if [[ -d "$data_dir" ]]; then
      cp -R "$data_dir" "$runtime_dir$(dirname "$data_dir")/"
    fi
  done

  # Supplying the MSVC redistributable here would hide missing wheel DLLs.
  bundled_msvc=$(find "$runtime_dir" -type f \( -iname 'msvcp*.dll' -o -iname 'vcruntime*.dll' \) -print)
  if [[ -n "$bundled_msvc" ]]; then
    printf 'ERROR: Git Bash staging would supply an MSVC runtime:\n%s\n' "$bundled_msvc" >&2
    exit 1
  fi

  cat > "$runtime_dir/run.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
export MSYSTEM=MINGW64
windows_root=$(/usr/bin/cygpath -u "${SYSTEMROOT:-${SystemRoot:?}}")
export PATH="/usr/bin:/mingw64/bin:${windows_root}/System32:${windows_root}:${windows_root}/System32/WindowsPowerShell/v1.0"
HOME=$(cygpath -u "${USERPROFILE:?}")
export HOME
mkdir -p "$HOME"
exec /usr/bin/bash --noprofile --norc "$@"
EOF

  runtime_native="${container_workspace}/$(basename "$runtime_dir")"
  platform_args+=(--isolation=process)
  payload=("${runtime_native}/usr/bin/bash.exe" --noprofile --norc "${runtime_native}/run.sh" "$script" "$@")
fi

# Forward only what the payload legitimately needs.
declare -a env_args=(
  # Only so pretty_printing.sh keeps emitting ::group:: markers.
  --env "GITHUB_ACTIONS=${GITHUB_ACTIONS:-}"
  # There is no nvcc in here; pin_cuda_toolkit falls back to this.
  --env "CCCL_CUDA_VERSION=${cuda_version}"
  # The v2 entry point sets this and then execs the v1 script, so this flag is
  # the only thing distinguishing a v2 run from a v1 one by the time we get here.
  --env "CCCL_PYTHON_USE_V2=${CCCL_PYTHON_USE_V2:-}"
  # nvidia-container-toolkit only injects libcuda.so.1 when `compute` is
  # requested. The nvidia/cuda images set this for you; a plain image does not.
  --env "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
  # Tells the payload it is the isolated side, and so must prove it (see
  # assert_minimal_environment in ci/pyenv_helper.sh).
  --env "CCCL_INSIDE_MINIMAL_CONTAINER=1"
  # The workspace is bind-mounted; do not litter it with the sibling's bytecode.
  --env "PYTHONDONTWRITEBYTECODE=1"
)

echo "::group::🐍 Running in minimal container: ${image}"
(
  set -x
  if $windows; then
    # Every Docker path is already native. Keep MSYS from rewriting mounts,
    # Windows executable paths, or arguments intended for the sibling's Bash.
    export MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL='*'
  fi
  docker run --rm -i \
    "${platform_args[@]}" \
    "${gpu_request[@]}" \
    "${env_args[@]}" \
    --mount "type=bind,source=${host_workspace},target=${container_workspace}" \
    --workdir "${container_workspace}" \
    "${image}" \
    "${payload[@]}"
)
echo "::endgroup::"
