# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate that a ``cuda-coop`` wheel contains the supported public layout.

The wheel is checked independently of imports so packaging regressions are
caught even when the source checkout would otherwise satisfy those imports.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import PurePosixPath

_PRIMITIVE_FAMILIES = {
    "adjacent_difference",
    "discontinuity",
    "exchange",
    "histogram",
    "load_store",
    "merge_sort",
    "radix",
    "reduce",
    "run_length_decode",
    "scan",
    "shuffle",
    "topk",
}

_PORTABLE_API_LEAVES = _PRIMITIVE_FAMILIES | {
    "temp_storage",
    "thread_data",
    "thread_group",
}

_REQUIRED_STUBS = {
    "cuda/coop/__init__.pyi",
    "cuda/coop/_typing.pyi",
    "cuda/coop/_core/api/__init__.pyi",
    *{f"cuda/coop/_core/api/{family}.pyi" for family in _PORTABLE_API_LEAVES},
    "cuda/coop/cutlass/__init__.pyi",
    "cuda/coop/cutlass/_temp_storage.pyi",
    "cuda/coop/cutlass/_thread_data.pyi",
    "cuda/coop/cutlass/_thread_group.pyi",
    "cuda/coop/cutlass/_typing.pyi",
    *{f"cuda/coop/cutlass/_group_{family}.pyi" for family in _PRIMITIVE_FAMILIES},
    "cuda/coop/numba_mlir/__init__.pyi",
    "cuda/coop/numba_mlir/_dataclass.pyi",
    "cuda/coop/numba_mlir/_enums.pyi",
    "cuda/coop/numba_mlir/_stateful_function.pyi",
    "cuda/coop/numba_mlir/_temp_storage.pyi",
    "cuda/coop/numba_mlir/_thread_data.pyi",
    "cuda/coop/numba_mlir/_thread_group.pyi",
    *{f"cuda/coop/numba_mlir/_group_{family}.pyi" for family in _PRIMITIVE_FAMILIES},
}

_OBSOLETE_LAYOUT_COMPONENTS = {"_block", "_dsl", "_internal", "_warp"}


def _reject_obsolete_layouts(names: set[str]) -> None:
    obsolete_members = sorted(
        name
        for name in names
        if name.startswith("cuda/coop/")
        and not name.startswith("cuda/coop/_headers/include/")
        and _OBSOLETE_LAYOUT_COMPONENTS.intersection(PurePosixPath(name).parts)
    )
    if obsolete_members:
        raise SystemExit(
            f"cuda-coop wheel contains obsolete package layouts: {obsolete_members}"
        )


def validate(wheel: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    _reject_obsolete_layouts(names)

    required = _REQUIRED_STUBS | {
        "cuda/coop/py.typed",
        "cuda/coop/_headers/cccl-bundle-provenance.json",
        "cuda/coop/_headers/include/cub/version.cuh",
        "cuda/coop/_headers/include/cub/block/block_load.cuh",
        "cuda/coop/_headers/include/cub/block/block_store.cuh",
        "cuda/coop/_headers/include/cuda/experimental/coop.cuh",
        "cuda/coop/_headers/include/cuda/experimental/group.cuh",
        "cuda/coop/_headers/include/thrust/detail/raw_pointer_cast.h",
        "cuda/coop/_headers/include/cuda/std/cstdint",
        "cuda/coop/_headers/include/nv/target",
    }
    missing = required - names
    if missing:
        raise SystemExit(
            f"cuda-coop wheel is missing required files: {sorted(missing)}"
        )
    if "cuda/__init__.py" in names:
        raise SystemExit(
            "cuda-coop wheel must not contain cuda/__init__.py; "
            "it would break the PEP 420 cuda namespace"
        )

    native_suffixes = {".a", ".dll", ".dylib", ".exe", ".lib", ".pyd", ".so"}
    native = sorted(
        name
        for name in names
        if any(
            suffix.lower() in native_suffixes for suffix in PurePosixPath(name).suffixes
        )
    )
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


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <cuda-coop-wheel>")
    validate(sys.argv[1])


if __name__ == "__main__":
    main()
