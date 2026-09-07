# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ....support.fixtures.cutlass_prims_runtime_source import (
    write_cutlass_api_row_sum_smoke,
)
from ..support.prims_runtime import (
    CUTLASS_API_MODULE,
    SOURCE_ROOT,
    _prims_runtime_env,
)


def _has_cub_row_reduce_headers() -> bool:
    return (
        SOURCE_ROOT.parents[1] / "cub" / "cub" / "block" / "block_row_reduce.cuh"
    ).is_file()


def _run_row_sum_api_smoke(
    tmp_path: Path,
    *,
    api_module: str,
) -> subprocess.CompletedProcess[str]:
    script_name = api_module.replace(".", "_")
    script_path = tmp_path / f"{script_name}_row_sum_smoke.py"
    write_cutlass_api_row_sum_smoke(
        script_path,
        source_root=SOURCE_ROOT,
        api_module=api_module,
    )
    result = subprocess.run(
        [sys.executable, "-B", str(script_path)],
        check=False,
        capture_output=True,
        env=_prims_runtime_env(tmp_path),
        text=True,
    )

    return result


def _run_cutlass_api_row_sum_smoke(
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_row_sum_api_smoke(
        tmp_path,
        api_module=CUTLASS_API_MODULE,
    )


@pytest.mark.skipif(
    not _has_cub_row_reduce_headers(),
    reason="requires CUB block_row_reduce.cuh",
)
def test_cutlass_api_row_sum_runtime(tmp_path: Path):
    result = _run_cutlass_api_row_sum_smoke(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"primitive_module": "cuda.coop.cutlass._block"' in result.stdout
    assert '"temp_storage_scope": "cuda.coop.cutlass._block"' in result.stdout
    assert '"row_total": 8128.0' in result.stdout
