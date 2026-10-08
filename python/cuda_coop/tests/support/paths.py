# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Find test, package and repository roots from this source file.

Compiler and subprocess tests need stable paths even when pytest starts from
another directory. Repository discovery checks both the CCCL build marker and
Python package directory, avoiding assumptions about the checkout's name.
"""

from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = TESTS_ROOT.parent


def _find_repo_root(start: Path) -> Path:
    """Walk upward to the CCCL checkout or fail with the starting path."""

    for candidate in (start, *start.parents):
        if (candidate / "CMakePresets.json").is_file() and (
            candidate / "python" / "cuda_coop"
        ).is_dir():
            return candidate
    raise RuntimeError(f"could not locate the CCCL repository above {start}")


REPO_ROOT = _find_repo_root(PACKAGE_ROOT)

__all__ = ["PACKAGE_ROOT", "REPO_ROOT", "TESTS_ROOT"]
