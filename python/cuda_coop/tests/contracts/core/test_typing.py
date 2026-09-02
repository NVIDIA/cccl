# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TYPING_FIXTURE = Path(__file__).parents[2] / "typing" / "root_api.py"
_PACKAGE_ROOT = Path(__file__).parents[3]


def test_public_root_stub_accepts_scalar_block_reduce(tmp_path: Path) -> None:
    mypy_api = pytest.importorskip("mypy.api")
    stdout, stderr, exit_status = mypy_api.run(
        [
            "--strict",
            "--config-file",
            str(_PACKAGE_ROOT / "pyproject.toml"),
            "--python-version",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            "--no-error-summary",
            "--cache-dir",
            str(tmp_path / "mypy-cache"),
            str(_TYPING_FIXTURE),
        ]
    )

    assert exit_status == 0, stdout + stderr
