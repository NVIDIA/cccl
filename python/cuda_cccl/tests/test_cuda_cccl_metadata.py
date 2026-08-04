# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).parents[1]


def test_metapackage_has_no_python_packages() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    assert metadata["tool"]["scikit-build"]["wheel"]["packages"] == []
    assert not (PROJECT_ROOT / "cuda").exists()


def test_exact_cuda_compute_dependency_matches_metapackage_version() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    project = metadata["project"]
    version = project["version"]

    assert project["dependencies"] == [f"cuda-compute=={version}"]
    assert "dynamic" not in project
    assert "dynamic-metadata" not in metadata["tool"]
    assert "setuptools_scm" not in metadata["tool"]


def test_all_cuda_compute_extras_are_forwarded() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    extras = metadata["project"]["optional-dependencies"]
    assert extras
    for extra, requirements in extras.items():
        assert requirements == [f"cuda-compute[{extra}]"]
