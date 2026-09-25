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
    assert project["description"] == (
        "Cooperative CUDA group primitives for Python DSLs"
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
        assert "cuda-pathfinder>=1.2.3" in requirements
        assert "numpy" in requirements
        assert "typing_extensions>=4.12.0" in requirements


def test_base_package_has_no_python_dependencies() -> None:
    assert _metadata()["project"]["dependencies"] == []


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


def test_python_implementation_boundaries() -> None:
    package = _PACKAGE_ROOT / "cuda" / "coop"
    required = (
        "cutlass/__init__.py",
        "cutlass/__init__.pyi",
        "cutlass/_compiler/_runtime.py",
        "cutlass/_thread_data.pyi",
        "cutlass/_group_exchange.py",
        "cutlass/_group_exchange.pyi",
        "cutlass/_group_merge_sort.py",
        "cutlass/_group_merge_sort.pyi",
        "cutlass/_group_radix.py",
        "cutlass/_group_radix.pyi",
        "cutlass/_group_reduce.py",
        "cutlass/_group_reduce.pyi",
        "cutlass/_group_scan.py",
        "cutlass/_group_scan.pyi",
        "cutlass/_group_shuffle.py",
        "cutlass/_group_shuffle.pyi",
        "cutlass/_lowering/_scan.py",
        "cutlass/_lowering/_exchange.py",
        "cutlass/_lowering/_merge_sort.py",
        "cutlass/_lowering/_radix.py",
        "cutlass/_lowering/_reduce.py",
        "cutlass/_lowering/_shuffle.py",
        "cutlass/_lowering/_thread_group.py",
        "cutlass/_temp_storage.py",
        "cutlass/_temp_storage.pyi",
        "cutlass/_compiler/_layout.py",
        "cutlass/_compiler/_storage.py",
        "numba_mlir/_lowering/_thread_group.py",
        "numba_mlir/_stateful_function.py",
        "numba_mlir/_stateful_function.pyi",
    )
    forbidden = (
        "_aot_cli.py",
        "numba_mlir/_dataclass.py",
        "numba_mlir/_scan_op.py",
    )

    missing = [relative for relative in required if not (package / relative).is_file()]
    assert not missing
    assert not [relative for relative in forbidden if (package / relative).exists()]

    warp_files = {
        path.relative_to(package / "_core" / "warp").as_posix()
        for path in (package / "_core" / "warp").rglob("*")
        if path.is_file() and path.suffix in {".py", ".pyi"}
    }
    assert warp_files == {
        "__init__.py",
        "exchange.py",
        "load_store.py",
        "merge_sort.py",
        "reduce.py",
        "reduce_batched.py",
        "scan.py",
    }
