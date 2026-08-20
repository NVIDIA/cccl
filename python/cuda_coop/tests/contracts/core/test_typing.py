# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

_TYPING_FIXTURES = Path(__file__).parents[2] / "typing"
_PACKAGE_ROOT = Path(__file__).parents[3]


@pytest.mark.parametrize("fixture_name", ("root_api.py", "cutlass_api.py"))
def test_public_stubs_accept_partial_tile_copy(
    fixture_name: str,
    tmp_path: Path,
) -> None:
    mypy_api = pytest.importorskip("mypy.api")
    fixture = _TYPING_FIXTURES / fixture_name
    stdout, stderr, exit_status = mypy_api.run(
        [
            "--strict",
            "--config-file",
            str(_PACKAGE_ROOT / "pyproject.toml"),
            "--no-error-summary",
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            str(fixture),
        ]
    )

    assert exit_status == 0, stdout + stderr
