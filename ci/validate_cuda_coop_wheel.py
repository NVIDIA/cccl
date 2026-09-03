# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate the supported layout of a standalone ``cuda-coop`` wheel.

The archive is checked without importing it so a source checkout or another
installed ``cuda`` namespace distribution cannot hide packaging regressions.
"""

from __future__ import annotations

import email.parser
import json
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

_REQUIRED_PACKAGE_FILES = {
    "cuda/coop/__init__.py",
    "cuda/coop/__init__.pyi",
    "cuda/coop/_typing.pyi",
    "cuda/coop/py.typed",
    "cuda/coop/_core/_auto_registration.py",
    "cuda/coop/_core/api/__init__.py",
    "cuda/coop/_core/api/__init__.pyi",
    "cuda/coop/_core/api/load_store.py",
    "cuda/coop/_core/api/load_store.pyi",
    "cuda/coop/_core/api/temp_storage.py",
    "cuda/coop/_core/api/temp_storage.pyi",
    "cuda/coop/_core/api/thread_data.py",
    "cuda/coop/_core/api/thread_data.pyi",
    "cuda/coop/_core/api/thread_group.py",
    "cuda/coop/_core/api/thread_group.pyi",
    "cuda/coop/_core/block/load_store.py",
    "cuda/coop/_core/group/load_store.py",
    "cuda/coop/_headers/_identity.py",
    "cuda/coop/_headers/_toolkit.py",
    "cuda/coop/numba_mlir/__init__.py",
    "cuda/coop/numba_mlir/__init__.pyi",
    "cuda/coop/numba_mlir/_compiler/_activation.py",
    "cuda/coop/numba_mlir/_compiler/_artifacts.py",
    "cuda/coop/numba_mlir/_compiler/_caching.py",
    "cuda/coop/numba_mlir/_compiler/_group_load_store.py",
    "cuda/coop/numba_mlir/_compiler/_group_planner.py",
    "cuda/coop/numba_mlir/_compiler/_nvrtc.py",
    "cuda/coop/numba_mlir/_compiler/_rewrite.py",
    "cuda/coop/numba_mlir/_enums.py",
    "cuda/coop/numba_mlir/_enums.pyi",
    "cuda/coop/numba_mlir/_group_load_store.py",
    "cuda/coop/numba_mlir/_group_load_store.pyi",
    "cuda/coop/numba_mlir/_lowering/_load_store.py",
    "cuda/coop/numba_mlir/_temp_storage.py",
    "cuda/coop/numba_mlir/_temp_storage.pyi",
    "cuda/coop/numba_mlir/_thread_data.py",
    "cuda/coop/numba_mlir/_thread_data.pyi",
    "cuda/coop/numba_mlir/_thread_group.py",
    "cuda/coop/numba_mlir/_thread_group.pyi",
    "cuda/coop/numba_mlir/py.typed",
}

_REQUIRED_HEADER_FILES = {
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

_REQUIRED_LICENSES = {
    "LICENSE",
    "cub/LICENSE.TXT",
    "cudax/LICENSE.TXT",
    "libcudacxx/LICENSE.TXT",
    "thrust/LICENSE",
}

_OBSOLETE_LAYOUT_COMPONENTS = {"_block", "_dsl", "_internal", "_warp"}
_NATIVE_SUFFIXES = {".a", ".dll", ".dylib", ".exe", ".lib", ".pyd", ".so"}
_FORBIDDEN_PACKAGE_FILES = {
    "cuda/coop/_aot_cli.py",
    "cuda/coop/_core/api/reduce.py",
    "cuda/coop/_core/api/reduce.pyi",
    "cuda/coop/_core/api/scan.py",
    "cuda/coop/_core/api/scan.pyi",
    "cuda/coop/_core/block/reduce.py",
    "cuda/coop/_core/block/scan.py",
    "cuda/coop/_core/group/reduce.py",
    "cuda/coop/_core/group/scan.py",
    "cuda/coop/numba_mlir/_dataclass.py",
    "cuda/coop/numba_mlir/_stateful_function.py",
    "cuda/coop/numba_mlir/_group_reduce.py",
    "cuda/coop/numba_mlir/_group_scan.py",
    "cuda/coop/numba_mlir/_lowering/_reduce.py",
    "cuda/coop/numba_mlir/_lowering/_scan.py",
    "cuda/coop/numba_mlir/_lowering/_thread_group.py",
    "cuda/coop/numba_mlir/_compiler/_rewrite_reduce.py",
    "cuda/coop/numba_mlir/_compiler/_rewrite_scan.py",
}


def _one_member(names: set[str], suffix: str) -> str:
    matches = sorted(name for name in names if name.endswith(suffix))
    if len(matches) != 1:
        raise SystemExit(
            f"cuda-coop wheel must contain exactly one {suffix}: {matches}"
        )
    return matches[0]


def _validate_metadata(archive: zipfile.ZipFile, names: set[str]) -> None:
    metadata_name = _one_member(names, ".dist-info/METADATA")
    wheel_name = _one_member(names, ".dist-info/WHEEL")

    metadata = email.parser.Parser().parsestr(
        archive.read(metadata_name).decode("utf-8")
    )
    if metadata.get("Name") != "cuda-coop":
        raise SystemExit(
            "cuda-coop wheel metadata has an unexpected project name: "
            f"{metadata.get('Name')!r}"
        )
    if metadata.get("Requires-Python") != ">=3.10":
        raise SystemExit(
            "cuda-coop wheel metadata must declare Requires-Python: >=3.10; got "
            f"{metadata.get('Requires-Python')!r}"
        )
    expected_extras = {
        "numba-cuda-mlir-cu12",
        "numba-cuda-mlir-cu13",
        "test",
    }
    extras = set(metadata.get_all("Provides-Extra", []))
    if extras != expected_extras:
        raise SystemExit(
            "cuda-coop wheel metadata has unexpected extras: "
            f"expected {sorted(expected_extras)}, got {sorted(extras)}"
        )
    if any(
        "cutlass" in requirement.lower()
        for requirement in metadata.get_all("Requires-Dist", [])
    ):
        raise SystemExit("cuda-coop wheel must not depend on CUTLASS")

    wheel_metadata = email.parser.Parser().parsestr(
        archive.read(wheel_name).decode("utf-8")
    )
    if wheel_metadata.get("Root-Is-Purelib", "").lower() != "true":
        raise SystemExit("cuda-coop wheel must declare Root-Is-Purelib: true")
    if wheel_metadata.get_all("Tag", []) != ["py3-none-any"]:
        raise SystemExit(
            "cuda-coop wheel must declare only Tag: py3-none-any; got "
            f"{wheel_metadata.get_all('Tag', [])}"
        )


def _validate_provenance(archive: zipfile.ZipFile) -> None:
    member = "cuda/coop/_headers/cccl-bundle-provenance.json"
    try:
        provenance = json.loads(archive.read(member))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SystemExit(f"cuda-coop header provenance is invalid JSON: {exc}") from exc
    if set(provenance) != {"cccl_source_commit"}:
        raise SystemExit(
            "cuda-coop header provenance must contain only cccl_source_commit"
        )
    revision = provenance["cccl_source_commit"]
    if not isinstance(revision, str) or (
        revision != "unknown" and re.fullmatch(r"[0-9a-f]{40}", revision) is None
    ):
        raise SystemExit("cuda-coop header provenance has an invalid source revision")


def validate(wheel: str | Path) -> None:
    wheel_path = Path(wheel)
    if not wheel_path.name.endswith("-py3-none-any.whl"):
        raise SystemExit(
            f"cuda-coop must produce a universal py3-none-any wheel: {wheel_path.name}"
        )

    with zipfile.ZipFile(wheel_path) as archive:
        names = set(archive.namelist())
        _validate_metadata(archive, names)

        missing = (_REQUIRED_PACKAGE_FILES | _REQUIRED_HEADER_FILES) - names
        if missing:
            raise SystemExit(
                f"cuda-coop wheel is missing required files: {sorted(missing)}"
            )

        if "cuda/__init__.py" in names:
            raise SystemExit(
                "cuda-coop wheel must not contain cuda/__init__.py; "
                "it would break the PEP 420 cuda namespace"
            )
        if any(name.startswith("cuda/coop/cutlass/") for name in names):
            raise SystemExit("cuda-coop wheel must not contain a CUTLASS backend")
        forbidden = sorted(_FORBIDDEN_PACKAGE_FILES & names)
        if forbidden:
            raise SystemExit(
                f"cuda-coop wheel contains excluded implementations: {forbidden}"
            )

        obsolete = sorted(
            name
            for name in names
            if name.startswith("cuda/coop/")
            and not name.startswith("cuda/coop/_headers/include/")
            and _OBSOLETE_LAYOUT_COMPONENTS.intersection(PurePosixPath(name).parts)
        )
        if obsolete:
            raise SystemExit(
                f"cuda-coop wheel contains obsolete package layouts: {obsolete}"
            )

        native = sorted(
            name
            for name in names
            if any(
                suffix.lower() in _NATIVE_SUFFIXES
                for suffix in PurePosixPath(name).suffixes
            )
        )
        if native:
            raise SystemExit(
                f"cuda-coop wheel must not contain native binaries: {native}"
            )

        license_members = {
            name.split(".dist-info/licenses/", 1)[1]
            for name in names
            if ".dist-info/licenses/" in name
        }
        missing_licenses = _REQUIRED_LICENSES - license_members
        if missing_licenses:
            raise SystemExit(
                "cuda-coop wheel is missing license payloads: "
                f"{sorted(missing_licenses)}"
            )

        _validate_provenance(archive)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <cuda-coop-wheel>")
    validate(sys.argv[1])


if __name__ == "__main__":
    main()
