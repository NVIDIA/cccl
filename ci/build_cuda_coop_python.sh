#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail

ci_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$ci_dir/.." && pwd)"

usage="Usage: $0 -py-version <python_version> [additional options...]"

# shellcheck source=ci/util/python/common_arg_parser.sh
source "$ci_dir/util/python/common_arg_parser.sh"
parse_python_args "$@"
require_py_version "$usage" || exit 1

# shellcheck source=ci/pyenv_helper.sh
source "$ci_dir/pyenv_helper.sh"
if [[ "${CCCL_CUDA_COOP_PYENV_READY:-0}" != "1" ]]; then
  setup_python_env "${py_version}" ".cccl-coop-build-venv"
fi

python -m pip install build

cd "$repo_root"
mkdir -p wheelhouse
rm -f wheelhouse/cuda_coop-*.whl
python -m build --wheel --outdir wheelhouse python/cuda_coop

mapfile -t wheels < <(find wheelhouse -maxdepth 1 -name 'cuda_coop-*.whl' -print | sort)
if [[ ${#wheels[@]} -ne 1 ]]; then
  echo "Error: expected exactly one cuda-coop wheel, found ${#wheels[@]}:" >&2
  printf '  %s\n' "${wheels[@]}" >&2
  exit 1
fi

if [[ "${wheels[0]}" != *-py3-none-any.whl ]]; then
  echo "Error: cuda-coop must produce a universal py3-none-any wheel: ${wheels[0]}" >&2
  exit 1
fi

python - "${wheels[0]}" <<'PY'
import sys
import zipfile
from pathlib import PurePosixPath

wheel = sys.argv[1]
with zipfile.ZipFile(wheel) as archive:
    names = set(archive.namelist())

required = {
    "cuda/coop/__init__.pyi",
    "cuda/coop/cutlass/__init__.pyi",
    "cuda/coop/cutlass/_load_store.pyi",
    "cuda/coop/cutlass/_thread_data.pyi",
    "cuda/coop/py.typed",
    "cuda/coop/_headers/cccl-bundle-provenance.json",
    "cuda/coop/_headers/include/cub/block/block_load.cuh",
    "cuda/coop/_headers/include/cub/block/block_store.cuh",
    "cuda/coop/_headers/include/thrust/detail/raw_pointer_cast.h",
    "cuda/coop/_headers/include/cuda/std/cstdint",
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
if any("cudax" in PurePosixPath(name).parts for name in names):
    raise SystemExit("cuda-coop wheel must not bundle CUDAX")
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
    "libcudacxx/LICENSE.TXT",
    "thrust/LICENSE",
}
missing_licenses = required_licenses - license_members
if missing_licenses:
    raise SystemExit(
        f"cuda-coop wheel is missing license payloads: {sorted(missing_licenses)}"
    )
PY

echo "Built cuda-coop wheel: ${wheels[0]}"

if [[ -n "${GITHUB_ACTIONS:-}" ]]; then
  wheel_artifact_name="$(CCCL_WHEEL_KIND=coop "$ci_dir/util/workflow/get_wheel_artifact_name.sh")"
  "$ci_dir/util/artifacts/upload.sh" "$wheel_artifact_name" 'wheelhouse/cuda_coop-.*\.whl'
fi
