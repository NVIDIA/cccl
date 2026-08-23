#!/usr/bin/env bash

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"

# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "Usage: $0 -py-version <python_version>" || exit 1

# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"
UV_VENV_CLEAR=1 setup_python_env "$py_version"

wheelhouse="${CUDA_COOP_WHEELHOUSE:-$repo_root/wheelhouse/cuda_coop}"
mkdir -p "$wheelhouse"
find "$wheelhouse" -maxdepth 1 -type f -name 'cuda_coop-*.whl' -delete

if [[ -z "${SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUDA_COOP:-}" ]]; then
  source_revision=$(git -C "$repo_root" rev-parse --short=12 HEAD)
  export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUDA_COOP="0.1.0.dev0+g${source_revision}"
fi

python -m pip install build
python -m build \
  --wheel \
  --outdir "$wheelhouse" \
  "$repo_root/python/cuda_coop"

coop_wheels=("$wheelhouse"/cuda_coop-*.whl)
if [[ ! -e "${coop_wheels[0]}" || ${#coop_wheels[@]} -ne 1 ]]; then
  echo "Error: expected one cuda_coop wheel under $wheelhouse" >&2
  exit 1
fi

if [[ "${coop_wheels[0]}" != *-py3-none-any.whl ]]; then
  echo "Error: cuda-coop must produce a universal py3-none-any wheel: ${coop_wheels[0]}" >&2
  exit 1
fi

python - "${coop_wheels[0]}" <<'PY'
import sys
import zipfile
from pathlib import PurePosixPath

wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())

required = {
    "cuda/coop/__init__.pyi",
    "cuda/coop/cutlass/__init__.pyi",
    "cuda/coop/cutlass/_types.pyi",
    "cuda/coop/cutlass/_group_load_store.pyi",
    "cuda/coop/cutlass/_block/__init__.pyi",
    "cuda/coop/cutlass/_warp/__init__.pyi",
    "cuda/coop/numba_mlir/__init__.pyi",
    "cuda/coop/numba_mlir/_block/__init__.pyi",
    "cuda/coop/numba_mlir/_warp/__init__.pyi",
    "cuda/coop/py.typed",
    "cuda/coop/_headers/cccl-bundle-provenance.json",
    "cuda/coop/_headers/include/cub/block/block_load.cuh",
    "cuda/coop/_headers/include/cub/block/block_store.cuh",
    "cuda/coop/_headers/include/thrust/detail/raw_pointer_cast.h",
    "cuda/coop/_headers/include/cuda/std/cstdint",
    "cuda/coop/_headers/include/cuda/experimental/coop.cuh",
    "cuda/coop/_headers/include/nv/target",
}
missing = required - names
if missing:
    raise SystemExit(f"cuda-coop wheel is missing required files: {sorted(missing)}")
if "cuda/__init__.py" in names:
    raise SystemExit(
        "cuda-coop wheel must not contain cuda/__init__.py; "
        "it would break the PEP 420 cuda namespace"
    )
native_suffixes = {".a", ".dll", ".dylib", ".exe", ".lib", ".pyd", ".so"}
native = sorted(name for name in names if PurePosixPath(name).suffix in native_suffixes)
if native:
    raise SystemExit(f"cuda-coop wheel must not contain native binaries: {native}")
license_members = {
    name.split(".dist-info/licenses/", 1)[1]
    for name in names
    if ".dist-info/licenses/" in name
}
required_licenses = {
    "LICENSE",
    "cub/LICENSE.TXT",
    "cudax/LICENSE.TXT",
    "libcudacxx/LICENSE.TXT",
    "thrust/LICENSE",
}
missing_licenses = required_licenses - license_members
if missing_licenses:
    raise SystemExit(
        f"cuda-coop wheel is missing license payloads: {sorted(missing_licenses)}"
    )
PY

echo "Built cuda-coop wheel: ${coop_wheels[0]}"

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  artifact_name=$(
    "$ci_dir/util/workflow/get_cuda_coop_wheel_artifact_name.sh" wheel
  )
  wheelhouse_relative=$(realpath --relative-to "$repo_root" "$wheelhouse")
  (
    cd "$repo_root"
    "$ci_dir/util/artifacts/upload.sh" \
      "$artifact_name" \
      "${wheelhouse_relative}/.*\\.whl"
  )
fi
