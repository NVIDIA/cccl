# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


def _load_stager() -> ModuleType:
    path = Path(__file__).parents[1] / "stage_python_wheel_publish.py"
    spec = importlib.util.spec_from_file_location("wheel_publish_stager", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_wheel(
    path: Path,
    distribution: str,
    version: str,
    requirements: tuple[str, ...] = (),
    payload: str = "same",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dist_info = f"{distribution.replace('-', '_')}-{version}.dist-info"
    metadata = f"Name: {distribution}\nVersion: {version}\n" + "".join(
        f"Requires-Dist: {requirement}\n" for requirement in requirements
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{distribution}/payload", payload)


def test_stages_only_the_metadata_selected_distribution(tmp_path: Path) -> None:
    stager = _load_stager()
    linux = tmp_path / "wheel-cccl-linux-amd64-py3.10"
    windows = tmp_path / "wheel-cccl-windows-amd64-py3.10"
    _write_wheel(
        linux / "cuda_compute-1.2.0-cp310-cp310-manylinux2014_x86_64.whl",
        "cuda-compute",
        "1.2.0",
    )
    _write_wheel(
        windows / "cuda_compute-1.2.0-cp310-cp310-win_amd64.whl",
        "cuda-compute",
        "1.2.0",
    )
    for artifact in (linux, windows):
        _write_wheel(
            artifact / "cuda_cccl-1.0.0-py3-none-any.whl",
            "cuda-cccl",
            "1.0.0",
            (
                "cuda-compute==1.2.0",
                'cuda-compute[cu12]==1.2.0; extra == "cu12"',
            ),
        )
    # Selection is based on METADATA identity, not this deceptive filename.
    _write_wheel(
        linux / "cuda_cccl_shadow-1.2.0-py3-none-any.whl",
        "unrelated-package",
        "1.2.0",
    )
    _write_wheel(
        tmp_path / "wheel-cccl-tsan-linux-amd64-py3.10/ignored.whl",
        "cuda-compute",
        "1.2.0",
    )

    destination = tmp_path / "publish"
    stager.stage_wheels(tmp_path, destination, "cuda-cccl", "1.0.0")

    assert [wheel.name for wheel in destination.glob("*.whl")] == [
        "cuda_cccl-1.0.0-py3-none-any.whl"
    ]
    assert (
        stager.exact_dependency_version(destination, "cuda-cccl", "cuda-compute")
        == "1.2.0"
    )

    compute_destination = tmp_path / "compute-publish"
    stager.stage_wheels(tmp_path, compute_destination, "cuda-compute", "1.2.0")
    assert sorted(wheel.name for wheel in compute_destination.glob("*.whl")) == [
        "cuda_compute-1.2.0-cp310-cp310-manylinux2014_x86_64.whl",
        "cuda_compute-1.2.0-cp310-cp310-win_amd64.whl",
    ]


def test_rejects_conflicting_duplicate_wheels(tmp_path: Path) -> None:
    stager = _load_stager()
    wheel_name = "cuda_cccl-1.0.0-py3-none-any.whl"
    _write_wheel(
        tmp_path / f"wheel-cccl-linux-amd64-py3.10/{wheel_name}",
        "cuda-cccl",
        "1.0.0",
        payload="linux",
    )
    _write_wheel(
        tmp_path / f"wheel-cccl-windows-amd64-py3.10/{wheel_name}",
        "cuda-cccl",
        "1.0.0",
        payload="windows",
    )

    with pytest.raises(RuntimeError, match="Conflicting wheels"):
        stager.stage_wheels(tmp_path, tmp_path / "publish", "cuda-cccl", "1.0.0")


def test_requires_selected_wheel_in_every_release_artifact(tmp_path: Path) -> None:
    stager = _load_stager()
    _write_wheel(
        tmp_path
        / "wheel-cccl-linux-amd64-py3.10"
        / "cuda_compute-1.2.0-cp310-cp310-manylinux2014_x86_64.whl",
        "cuda-compute",
        "1.2.0",
    )
    _write_wheel(
        tmp_path
        / "wheel-cccl-windows-amd64-py3.10"
        / "cuda_cccl-1.0.0-py3-none-any.whl",
        "cuda-cccl",
        "1.0.0",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Expected one cuda-compute 1.2.0 wheel in wheel-cccl-windows-amd64-py3.10"
        ),
    ):
        stager.stage_wheels(tmp_path, tmp_path / "publish", "cuda-compute", "1.2.0")


@pytest.mark.parametrize(
    "requirement",
    (
        "cuda-compute>=1.2.0",
        "cuda-compute==1.2.0,!=1.2.1",
        'cuda-compute==1.2.0; python_version > "3.10"',
    ),
)
def test_requires_one_exact_unconditional_dependency(
    tmp_path: Path, requirement: str
) -> None:
    stager = _load_stager()
    wheelhouse = tmp_path / "publish"
    _write_wheel(
        wheelhouse / "cuda_cccl-1.0.0-py3-none-any.whl",
        "cuda-cccl",
        "1.0.0",
        (requirement,),
    )

    with pytest.raises(RuntimeError, match="exact"):
        stager.exact_dependency_version(wheelhouse, "cuda-cccl", "cuda-compute")


def test_rejects_other_versions_and_non_public_tags(tmp_path: Path) -> None:
    stager = _load_stager()
    _write_wheel(
        tmp_path
        / "wheel-cccl-linux-amd64-py3.10"
        / "cuda_compute-1.2.1-cp310-cp310-manylinux2014_x86_64.whl",
        "cuda-compute",
        "1.2.1",
    )

    with pytest.raises(RuntimeError, match="Expected one cuda-compute 1.2.0"):
        stager.stage_wheels(tmp_path, tmp_path / "publish", "cuda-compute", "1.2.0")
    with pytest.raises(RuntimeError, match="canonical public PEP 440"):
        stager.release_tag("cuda-compute", "1.2.0+local")
