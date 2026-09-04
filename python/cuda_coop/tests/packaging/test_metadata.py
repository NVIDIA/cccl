# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

_PACKAGE_ROOT = Path(__file__).parents[2]


def _metadata() -> dict[str, Any]:
    with (_PACKAGE_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_project_metadata_declares_the_supported_python_range() -> None:
    project = _metadata()["project"]

    assert project["name"] == "cuda-coop"
    assert (
        project["description"]
        == "Cooperative CUDA Block and Warp Load and Store for Python DSLs"
    )
    assert project["requires-python"] == ">=3.10"
    assert set(project["classifiers"]) >= {
        f"Programming Language :: Python :: 3.{minor}" for minor in range(10, 15)
    }


def test_only_numba_cuda_mlir_backend_extras_are_published() -> None:
    optional = _metadata()["project"]["optional-dependencies"]

    assert set(optional) == {
        "numba-cuda-mlir-cu12",
        "numba-cuda-mlir-cu13",
        "test",
    }
    for cuda_major in (12, 13):
        requirements = optional[f"numba-cuda-mlir-cu{cuda_major}"]
        assert "cuda-core>=0.5.1,<2" in requirements
        assert f"numba-cuda-mlir[cu{cuda_major}]>=0.5.0,<0.6" in requirements


def test_build_metadata_requires_a_universal_wheel() -> None:
    scikit_build = _metadata()["tool"]["scikit-build"]

    assert scikit_build["wheel"]["py-api"] == "py3"
    assert scikit_build["wheel"]["platlib"] is False
    assert scikit_build["wheel"]["packages"] == {"cuda/coop": "cuda/coop"}


def test_all_bundled_header_licenses_are_declared() -> None:
    force_include = _metadata()["tool"]["scikit-build"]["wheel"]["force-include"]

    assert force_include == {
        "../../LICENSE": "${SKBUILD_METADATA_DIR}/licenses/LICENSE",
        "../../cub/LICENSE.TXT": ("${SKBUILD_METADATA_DIR}/licenses/cub/LICENSE.TXT"),
        "../../cudax/LICENSE.TXT": (
            "${SKBUILD_METADATA_DIR}/licenses/cudax/LICENSE.TXT"
        ),
        "../../libcudacxx/LICENSE.TXT": (
            "${SKBUILD_METADATA_DIR}/licenses/libcudacxx/LICENSE.TXT"
        ),
        "../../thrust/LICENSE": ("${SKBUILD_METADATA_DIR}/licenses/thrust/LICENSE"),
    }


def test_excluded_python_implementations_are_absent() -> None:
    package = _PACKAGE_ROOT / "cuda" / "coop"
    forbidden = (
        "_aot_cli.py",
        "cutlass",
        "_core/block/reduce.py",
        "_core/block/scan.py",
        "_core/group/reduce.py",
        "_core/group/scan.py",
        "_core/api/reduce.py",
        "_core/api/reduce.pyi",
        "_core/api/scan.py",
        "_core/api/scan.pyi",
        "numba_mlir/_dataclass.py",
        "numba_mlir/_stateful_function.py",
        "numba_mlir/_group_reduce.py",
        "numba_mlir/_group_scan.py",
        "numba_mlir/_lowering/_reduce.py",
        "numba_mlir/_lowering/_scan.py",
        "numba_mlir/_lowering/_thread_group.py",
        "numba_mlir/_compiler/_rewrite_reduce.py",
        "numba_mlir/_compiler/_rewrite_scan.py",
    )

    assert not [relative for relative in forbidden if (package / relative).exists()]

    warp_files = {
        path.relative_to(package / "_core" / "warp").as_posix()
        for path in (package / "_core" / "warp").rglob("*")
        if path.is_file() and path.suffix in {".py", ".pyi"}
    }
    assert warp_files == {"__init__.py", "load_store.py"}
