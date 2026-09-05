# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ....support.fixtures.cutlass_prims_runtime_source import (
    write_prims_api_vector_block_prefix_segment_smoke,
)
from ..support.prims_runtime import (
    CUTLASS_API_MODULE,
    SOURCE_ROOT,
    _prims_runtime_env,
    _run_prims_example,
)


def _run_block_prefix_segment_api_smoke(
    tmp_path: Path,
    *,
    api_module: str,
    call_mode: str,
) -> subprocess.CompletedProcess[str]:
    script_name = api_module.replace(".", "_")
    script_path = (
        tmp_path / f"{script_name}_vector_block_prefix_segment_{call_mode}_smoke.py"
    )
    write_prims_api_vector_block_prefix_segment_smoke(
        script_path,
        source_root=SOURCE_ROOT,
        api_module=api_module,
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


def _run_cutlass_api_block_prefix_segment_factory_smoke(
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_block_prefix_segment_api_smoke(
        tmp_path,
        api_module=CUTLASS_API_MODULE,
        call_mode="factory",
    )


def _run_cutlass_api_block_prefix_segment_direct_smoke(
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_block_prefix_segment_api_smoke(
        tmp_path,
        api_module=CUTLASS_API_MODULE,
        call_mode="direct",
    )


def test_prims_vector_block_prefix_segment_example_runtime(tmp_path: Path):
    result = _run_prims_example("prims_vector_block_prefix_segment.py", tmp_path)

    assert result.returncode == 0, result.stderr
    assert "'exclusive':" in result.stdout
    assert "'inclusive':" in result.stdout
    assert "'sum':" in result.stdout
    assert "'xor':" in result.stdout
    assert "'diff_right':" in result.stdout
    assert "'heads':" in result.stdout
    assert "'tails':" in result.stdout
    assert "'shuffle_down':" in result.stdout
    assert "'shuffle_prefix':" in result.stdout


def test_cutlass_api_prims_vector_block_prefix_segment_factory_runtime(tmp_path: Path):
    result = _run_cutlass_api_block_prefix_segment_factory_smoke(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"factory_scopes":' in result.stdout
    assert result.stdout.count('"cuda.coop.cutlass._block"') == 8
    assert '"exclusive":' in result.stdout
    assert '"inclusive":' in result.stdout
    assert '"xor_prefix":' in result.stdout
    assert '"sum_aggregate":' in result.stdout
    assert '"xor_aggregate":' in result.stdout
    assert '"sum":' in result.stdout
    assert '"xor":' in result.stdout
    assert '"diff_right":' in result.stdout
    assert '"heads":' in result.stdout
    assert '"tails":' in result.stdout
    assert '"shuffle_down":' in result.stdout
    assert '"shuffle_prefix":' in result.stdout


def test_cutlass_api_prims_vector_block_prefix_segment_direct_runtime(tmp_path: Path):
    result = _run_cutlass_api_block_prefix_segment_direct_smoke(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"primitive_modules":' in result.stdout
    assert result.stdout.count('"cuda.coop.cutlass._block"') == 9
    assert '"exclusive":' in result.stdout
    assert '"inclusive":' in result.stdout
    assert '"xor_prefix":' in result.stdout
    assert '"sum_aggregate":' in result.stdout
    assert '"xor_aggregate":' in result.stdout
    assert '"sum":' in result.stdout
    assert '"xor":' in result.stdout
    assert '"diff_right":' in result.stdout
    assert '"heads":' in result.stdout
    assert '"tails":' in result.stdout
    assert '"shuffle_down":' in result.stdout
    assert '"shuffle_prefix":' in result.stdout
