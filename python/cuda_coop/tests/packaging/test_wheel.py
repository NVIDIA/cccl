# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import json
import os
import re
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

import pytest

_REPO_ROOT = Path(__file__).parents[4]
_WHEEL_ENVIRONMENT_VARIABLE = "CUDA_COOP_TEST_WHEEL"

_REQUIRED_PACKAGE_MEMBERS = {
    "cuda/coop/__init__.py",
    "cuda/coop/__init__.pyi",
    "cuda/coop/_typing.pyi",
    "cuda/coop/py.typed",
    "cuda/coop/_core/api/__init__.pyi",
    "cuda/coop/_core/api/load_store.pyi",
    "cuda/coop/_core/api/temp_storage.pyi",
    "cuda/coop/_core/api/thread_data.pyi",
    "cuda/coop/_core/api/thread_group.pyi",
    "cuda/coop/numba_mlir/__init__.py",
    "cuda/coop/numba_mlir/__init__.pyi",
    "cuda/coop/numba_mlir/_enums.py",
    "cuda/coop/numba_mlir/_enums.pyi",
    "cuda/coop/numba_mlir/_group_load_store.py",
    "cuda/coop/numba_mlir/_group_load_store.pyi",
    "cuda/coop/numba_mlir/_temp_storage.py",
    "cuda/coop/numba_mlir/_temp_storage.pyi",
    "cuda/coop/numba_mlir/_thread_data.py",
    "cuda/coop/numba_mlir/_thread_data.pyi",
    "cuda/coop/numba_mlir/_thread_group.py",
    "cuda/coop/numba_mlir/_thread_group.pyi",
}

_REQUIRED_HEADER_MEMBERS = {
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

_FORBIDDEN_PACKAGE_MEMBERS = {
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


def _wheel_under_test() -> Path:
    configured = os.environ.get(_WHEEL_ENVIRONMENT_VARIABLE)
    if configured:
        wheel = Path(configured).resolve()
        assert wheel.is_file(), f"configured wheel does not exist: {wheel}"
        return wheel

    wheels = sorted((_REPO_ROOT / "wheelhouse").glob("cuda_coop-*.whl"))
    if not wheels:
        pytest.skip(
            f"set {_WHEEL_ENVIRONMENT_VARIABLE} or build a wheel under wheelhouse/"
        )
    assert len(wheels) == 1, f"expected one cuda-coop wheel, found {wheels}"
    return wheels[0]


def _single_member(names: set[str], suffix: str) -> str:
    matches = sorted(name for name in names if name.endswith(suffix))
    assert len(matches) == 1, f"expected one {suffix!r} member, found {matches}"
    return matches[0]


def test_wheel_is_universal_and_contains_the_complete_payload() -> None:
    wheel = _wheel_under_test()

    assert wheel.name.endswith("-py3-none-any.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())

        missing = (_REQUIRED_PACKAGE_MEMBERS | _REQUIRED_HEADER_MEMBERS) - names
        assert not missing, f"wheel is missing required members: {sorted(missing)}"

        forbidden = _FORBIDDEN_PACKAGE_MEMBERS & names
        assert not forbidden, f"wheel contains excluded implementations: {forbidden}"
        assert not any(name.startswith("cuda/coop/cutlass/") for name in names)
        assert not any(name.startswith("cuda/coop/_core/warp/") for name in names)
        assert "cuda/__init__.py" not in names

        native_suffixes = {".a", ".dll", ".dylib", ".exe", ".lib", ".pyd", ".so"}
        native = sorted(
            name
            for name in names
            if any(
                suffix.lower() in native_suffixes
                for suffix in PurePosixPath(name).suffixes
            )
        )
        assert not native, f"universal wheel contains native binaries: {native}"

        metadata_name = _single_member(names, ".dist-info/METADATA")
        metadata = BytesParser(policy=policy.default).parsebytes(
            archive.read(metadata_name)
        )
        assert metadata["Name"] == "cuda-coop"
        assert metadata["Requires-Python"] == ">=3.10"
        assert set(metadata.get_all("Provides-Extra", [])) == {
            "numba-cuda-mlir-cu12",
            "numba-cuda-mlir-cu13",
            "test",
        }
        assert not any(
            "cutlass" in requirement.lower()
            for requirement in metadata.get_all("Requires-Dist", [])
        )

        wheel_metadata_name = _single_member(names, ".dist-info/WHEEL")
        wheel_metadata = BytesParser(policy=policy.default).parsebytes(
            archive.read(wheel_metadata_name)
        )
        assert wheel_metadata["Root-Is-Purelib"] == "true"
        assert wheel_metadata.get_all("Tag") == ["py3-none-any"]

        provenance = json.loads(
            archive.read("cuda/coop/_headers/cccl-bundle-provenance.json").decode(
                "utf-8"
            )
        )
        assert set(provenance) == {"cccl_source_commit"}
        assert provenance["cccl_source_commit"] == "unknown" or re.fullmatch(
            r"[0-9a-f]{40}", provenance["cccl_source_commit"]
        )

        license_members = {
            name.split(".dist-info/licenses/", 1)[1]
            for name in names
            if ".dist-info/licenses/" in name
        }
        assert license_members >= {
            "LICENSE",
            "cub/LICENSE.TXT",
            "cudax/LICENSE.TXT",
            "libcudacxx/LICENSE.TXT",
            "thrust/LICENSE",
        }
