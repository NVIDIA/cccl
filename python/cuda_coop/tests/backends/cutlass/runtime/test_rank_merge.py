# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ....support.fixtures.cutlass_prims_runtime_source import (
    write_prims_api_vector_rank_merge_smoke,
)
from ..support.prims_runtime import (
    CUTLASS_API_MODULE,
    SOURCE_ROOT,
    _prims_runtime_env,
)


def _run_block_rank_merge_api_smoke(
    tmp_path: Path,
    *,
    call_mode: str,
) -> subprocess.CompletedProcess[str]:
    script_name = CUTLASS_API_MODULE.replace(".", "_")
    script_path = tmp_path / f"{script_name}_vector_rank_merge_{call_mode}_smoke.py"
    write_prims_api_vector_rank_merge_smoke(
        script_path,
        source_root=SOURCE_ROOT,
        api_module=CUTLASS_API_MODULE,
        call_mode=call_mode,
    )
    result = subprocess.run(
        [sys.executable, "-B", str(script_path)],
        check=False,
        capture_output=True,
        env=_prims_runtime_env(tmp_path),
        text=True,
    )

    return result


def _run_cutlass_api_rank_merge_factory_smoke(
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_block_rank_merge_api_smoke(
        tmp_path,
        call_mode="factory",
    )


def _run_cutlass_api_rank_merge_direct_smoke(
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_block_rank_merge_api_smoke(
        tmp_path,
        call_mode="direct",
    )


def test_cutlass_api_prims_vector_rank_merge_factory_runtime(tmp_path: Path):
    result = _run_cutlass_api_rank_merge_factory_smoke(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"factory_scopes":' in result.stdout
    assert result.stdout.count('"cuda.coop.cutlass._block"') == 3
    assert '"merge_pairs":' in result.stdout
    assert '"prefix":' in result.stdout
    assert '"ranks":' in result.stdout


def test_cutlass_api_prims_vector_rank_merge_direct_runtime(tmp_path: Path):
    result = _run_cutlass_api_rank_merge_direct_smoke(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"primitive_modules":' in result.stdout
    assert result.stdout.count('"cuda.coop.cutlass._block"') == 4
    assert '"merge_pairs":' in result.stdout
    assert '"prefix":' in result.stdout
    assert '"ranks":' in result.stdout
