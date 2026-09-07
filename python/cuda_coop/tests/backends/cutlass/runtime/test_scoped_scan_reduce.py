# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ....support.fixtures.cutlass_prims_runtime_source import (
    write_prims_api_vector_float64_scan_reduce_smoke,
)
from ..support.prims_runtime import (
    CUTLASS_API_MODULE,
    SOURCE_ROOT,
    _prims_runtime_env,
    _run_prims_example,
)


def _run_float64_scan_reduce_api_smoke(
    tmp_path: Path,
    *,
    api_module: str,
) -> subprocess.CompletedProcess[str]:
    script_name = api_module.replace(".", "_")
    script_path = tmp_path / f"{script_name}_vector_float64_scan_reduce_smoke.py"
    write_prims_api_vector_float64_scan_reduce_smoke(
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


def _run_cutlass_api_float64_scan_reduce_smoke(
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_float64_scan_reduce_api_smoke(
        tmp_path,
        api_module=CUTLASS_API_MODULE,
    )


def test_mixed_tensor_vector_scan_example_runtime(tmp_path: Path):
    result = _run_prims_example("mixed_tensor_vector_scan.py", tmp_path)

    assert result.returncode == 0, result.stderr
    assert "'tensor_prefix':" in result.stdout
    assert "'vector_prefix':" in result.stdout


def test_cutlass_api_prims_vector_float64_scan_reduce_runtime(tmp_path: Path):
    result = _run_cutlass_api_float64_scan_reduce_smoke(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"primitive_modules":' in result.stdout
    assert result.stdout.count('"cuda.coop.cutlass._block"') == 5
    assert '"exclusive":' in result.stdout
    assert '"inclusive":' in result.stdout
    assert '"sum":' in result.stdout
    assert '"reduce":' in result.stdout
