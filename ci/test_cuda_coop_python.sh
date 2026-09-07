#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"
source "$ci_dir/pyenv_helper.sh"

# Parse common arguments
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "Usage: $0 -py-version <python_version> [-stage <stage>] [--dry-run]"

stages=()
dry_run=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    -stage | --stage)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value" >&2
        exit 1
      fi
      stages+=("$2")
      shift 2
      ;;
    -stage=* | --stage=*)
      stages+=("${1#*=}")
      shift
      ;;
    --dry-run)
      dry_run=true
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# Default to the backend-free contract lane. Compiler-backed lanes select their
# isolated dependency environment explicitly.
if [[ ${#stages[@]} -eq 0 ]]; then
  stages=("contracts")
fi

for stage in "${stages[@]}"; do
  python "$ci_dir/util/python/cuda_coop_test_driver.py" \
    --repo-root "$repo_root" \
    --stage "$stage" \
    --dry-run
done

if [[ "$dry_run" == true ]]; then
  exit 0
fi

needs_cuda_toolkit=false
for stage in "${stages[@]}"; do
  if [[ "$stage" != "contracts" ]]; then
    needs_cuda_toolkit=true
  fi
done

# Setup Python environment
UV_VENV_CLEAR=1 setup_python_env "${py_version}"
cuda_major_version=""
if [[ "$needs_cuda_toolkit" == true ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "Error: the selected cuda.coop stage requires nvcc on PATH" >&2
    exit 1
  fi
  pin_cuda_toolkit
fi

reject_cuda_cccl() {
  if python - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    installed = version("cuda-cccl")
except PackageNotFoundError:
    raise SystemExit(1)
print(installed)
PY
  then
    echo "Error: cuda-coop CI must not install the legacy cuda-cccl distribution" >&2
    exit 1
  fi
}

reject_cuda_cccl

coop_extras=()
needs_cutlass_runtime=false
needs_numba_mlir_runtime=false
add_coop_extra() {
  local candidate="$1"
  local existing
  for existing in "${coop_extras[@]}"; do
    if [[ "$existing" == "$candidate" ]]; then
      return
    fi
  done
  coop_extras+=("$candidate")
}

for stage in "${stages[@]}"; do
  case "$stage" in
    contracts)
      add_coop_extra "test"
      ;;
    numba-mlir | numba-mlir-host | numba-mlir-qualification | numba-mlir-cluster-qualification)
      needs_numba_mlir_runtime=true
      add_coop_extra "cu${cuda_major_version}"
      add_coop_extra "test"
      ;;
    cutlass)
      if [[ "$cuda_major_version" != "13" ]]; then
        echo "Error: CUTLASS stages support CUDA 13 only" >&2
        exit 1
      fi
      needs_cutlass_runtime=true
      needs_numba_mlir_runtime=true
      add_coop_extra "cu13"
      add_coop_extra "examples"
      add_coop_extra "test"
      ;;
    cutlass-host | cutlass-final-link-qualification | cutlass-cluster-qualification | cutlass-sm100-qualification)
      if [[ "$cuda_major_version" != "13" ]]; then
        echo "Error: CUTLASS stages support CUDA 13 only" >&2
        exit 1
      fi
      needs_cutlass_runtime=true
      add_coop_extra "cu13"
      add_coop_extra "examples"
      add_coop_extra "test"
      ;;
    *)
      echo "Error: unknown cuda.coop test stage '$stage'" >&2
      exit 1
      ;;
  esac
done

joined_extras=$(IFS=,; echo "${coop_extras[*]}")
local_wheelhouse=$(mktemp -d -t cuda-coop-wheelhouse.XXXXXX)
if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  producer_id=$("$ci_dir/util/workflow/get_producer_id.sh")
  artifact_name=$(
    "$ci_dir/util/workflow/get_cuda_coop_wheel_artifact_name.sh" \
      wheel \
      "$producer_id"
  )
  "$ci_dir/util/artifacts/download.sh" "$artifact_name" "$local_wheelhouse"
else
  CUDA_COOP_WHEELHOUSE="$local_wheelhouse" \
    "$ci_dir/build_cuda_coop_python.sh" -py-version "$py_version"
fi

cuda_coop_wheels=("$local_wheelhouse"/cuda_coop-*.whl)
if [[ ! -e "${cuda_coop_wheels[0]}" ]]; then
  echo "Error: cuda_coop wheel not found under $local_wheelhouse" >&2
  exit 1
fi
if [[ ${#cuda_coop_wheels[@]} -ne 1 ]]; then
  echo "Error: expected one cuda_coop wheel, found ${#cuda_coop_wheels[@]}" >&2
  exit 1
fi
python -m pip install \
  --find-links "$local_wheelhouse" \
  "${cuda_coop_wheels[0]}[$joined_extras]"

# Compiler runtimes install from public indexes through the wheel extras by
# default. Qualification lanes may pin exact compiler artifacts instead by
# exporting CUDA_COOP_CUTLASS_REQUIREMENTS_FILE (optionally with
# CUDA_COOP_CUTLASS_EXTRA_INDEX_URL) or CUDA_COOP_NUMBA_MLIR_REQUIREMENTS_FILE.
if [[ "$needs_cutlass_runtime" == true ]]; then
  if [[ -n "${CUDA_COOP_CUTLASS_REQUIREMENTS_FILE:-}" ]]; then
    if [[ ! -r "$CUDA_COOP_CUTLASS_REQUIREMENTS_FILE" ]]; then
      echo \
        "Error: CUDA_COOP_CUTLASS_REQUIREMENTS_FILE is not a readable file: $CUDA_COOP_CUTLASS_REQUIREMENTS_FILE" \
        >&2
      exit 1
    fi
    python -m pip install \
      ${CUDA_COOP_CUTLASS_EXTRA_INDEX_URL:+--extra-index-url "$CUDA_COOP_CUTLASS_EXTRA_INDEX_URL"} \
      --requirement "$CUDA_COOP_CUTLASS_REQUIREMENTS_FILE"
  else
    python -m pip install "${cuda_coop_wheels[0]}[cutlass]"
  fi
fi

if [[ "$needs_numba_mlir_runtime" == true ]]; then
  if [[ -n "${CUDA_COOP_NUMBA_MLIR_REQUIREMENTS_FILE:-}" ]]; then
    if [[ ! -r "$CUDA_COOP_NUMBA_MLIR_REQUIREMENTS_FILE" ]]; then
      echo \
        "Error: CUDA_COOP_NUMBA_MLIR_REQUIREMENTS_FILE is not a readable file: $CUDA_COOP_NUMBA_MLIR_REQUIREMENTS_FILE" \
        >&2
      exit 1
    fi
    python -m pip install \
      --requirement "$CUDA_COOP_NUMBA_MLIR_REQUIREMENTS_FILE"
  else
    python -m pip install "${cuda_coop_wheels[0]}[numba-cuda-mlir-cu${cuda_major_version}]"
  fi
fi

reject_cuda_cccl
python -m pip check

if [[ "$needs_cuda_toolkit" == true ]]; then
  echo "nvcc provenance:"
  nvcc --version
else
  echo "nvcc provenance: not required for backend-free contracts"
fi

for stage in "${stages[@]}"; do
  python "$ci_dir/util/python/cuda_coop_test_driver.py" \
    --repo-root "$repo_root" \
    --stage "$stage"
done
