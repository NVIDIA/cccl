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
# The existing cuda_cccl source directory now builds the cuda-compute distribution.
COMPUTE_PYPROJECT = PROJECT_ROOT.parent / "cuda_cccl" / "pyproject.toml"


def test_metapackage_contract() -> None:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        metadata = tomllib.load(stream)
    assert COMPUTE_PYPROJECT.is_file(), (
        f"cuda-compute project metadata not found at {COMPUTE_PYPROJECT}"
    )
    with COMPUTE_PYPROJECT.open("rb") as stream:
        compute_metadata = tomllib.load(stream)

    assert metadata["tool"]["scikit-build"]["wheel"]["packages"] == []
    assert not (PROJECT_ROOT / "cuda").exists()
    project = metadata["project"]
    version = project["version"]
    assert version == compute_metadata["project"]["version"]
    assert project["dependencies"] == [f"cuda-compute=={version}"]
    extras = metadata["project"]["optional-dependencies"]
    compute_extras = compute_metadata["project"]["optional-dependencies"]
    assert extras
    assert set(extras) == set(compute_extras)
    for extra, requirements in extras.items():
        assert requirements == [f"cuda-compute[{extra}]"]
