# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import sys
import zipfile
from pathlib import PurePosixPath


def validate(wheel: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

    required = {
        "cuda/coop/__init__.pyi",
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
        name for name in names if PurePosixPath(name).suffix in native_suffixes
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
