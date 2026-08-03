# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import tomllib

PROJECT_ROOT = Path(__file__).parents[1]


def test_metapackage_has_no_python_packages() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    assert metadata["tool"]["scikit-build"]["wheel"]["packages"] == []
    assert not (PROJECT_ROOT / "cuda").exists()


def test_exact_cuda_compute_dependency_follows_metapackage_version() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    project = metadata["project"]
    providers = metadata["tool"]["dynamic-metadata"]

    assert project["dependencies"] == []
    assert project["dynamic"] == ["version", "dependencies"]
    assert providers == [
        {"provider": "scikit_build_core.metadata.setuptools_scm"},
        {
            "provider": "scikit_build_core.metadata.template",
            "field": "dependencies",
            "result": ["cuda-compute=={project[version]}"],
        },
    ]


def test_all_cuda_compute_extras_are_forwarded() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)

    extras = metadata["project"]["optional-dependencies"]
    assert extras
    for extra, requirements in extras.items():
        assert requirements == [f"cuda-compute[{extra}]"]
