# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).parents[2]


def _load_merger() -> ModuleType:
    path = PROJECT_ROOT / "merge_cuda_wheels.py"
    spec = importlib.util.spec_from_file_location("cuda_compute_wheel_merger", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_extracted_wheel(root: Path, cuda_major: int, shared: str) -> Path:
    (root / "cuda/compute").mkdir(parents=True)
    (root / "cuda/compute/__init__.py").write_text(shared)
    (root / "cuda/compute/_hostjit/clang").mkdir(parents=True)
    (root / "cuda/compute/_hostjit/clang/header.h").write_text(shared)
    (root / f"cuda/compute/cu{cuda_major}").mkdir(parents=True)
    (root / f"cuda/compute/cu{cuda_major}/_bindings.so").write_text(f"cu{cuda_major}")
    (root / "cuda_compute-1.0.dist-info").mkdir()
    (root / "cuda_compute-1.0.dist-info/METADATA").write_text(shared)
    (root / "cuda_compute-1.0.dist-info/RECORD").write_text(f"cu{cuda_major}")
    return root


def test_shared_payload_may_match_while_cuda_payload_differs(tmp_path: Path) -> None:
    merger = _load_merger()
    cu12 = _make_extracted_wheel(tmp_path / "cu12", 12, "shared")
    cu14 = _make_extracted_wheel(tmp_path / "cu14", 14, "shared")

    merger._validate_shared_contents(
        [cu12, cu14],
        [Path("cuda_compute.cu12.whl"), Path("cuda_compute.cu14.whl")],
    )


def test_divergent_shared_payload_is_rejected(tmp_path: Path) -> None:
    merger = _load_merger()
    cu12 = _make_extracted_wheel(tmp_path / "cu12", 12, "shared")
    cu13 = _make_extracted_wheel(tmp_path / "cu13", 13, "shared")
    divergent_header = cu13 / "cuda/compute/_hostjit/clang/header.h"
    divergent_header.write_text("divergent")

    with pytest.raises(
        RuntimeError,
        match=(
            r"non-CUDA-major wheel contents differ "
            r"\(missing=\[\], extra=\[\], "
            r"changed=\['cuda/compute/_hostjit/clang/header.h'\]\)"
        ),
    ):
        merger._validate_shared_contents(
            [cu12, cu13],
            [Path("cuda_compute.cu12.whl"), Path("cuda_compute.cu13.whl")],
        )
