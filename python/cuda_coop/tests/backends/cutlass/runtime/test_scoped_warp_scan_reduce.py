# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ....support.fixtures.cutlass_prims_runtime_source import (
    write_cutlass_api_vector_warp_factory_smoke,
    write_prims_api_vector_warp_prefix_smoke,
)
from ..support.prims_runtime import (
    CUTLASS_API_MODULE,
    SOURCE_ROOT,
    _prims_runtime_env,
    _run_prims_example,
)


def _run_warp_api_smoke(
    tmp_path: Path,
    *,
    api_module: str,
) -> subprocess.CompletedProcess[str]:
    script_name = api_module.replace(".", "_")
    script_path = tmp_path / f"{script_name}_vector_warp_prefix_smoke.py"
    write_prims_api_vector_warp_prefix_smoke(
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


def _run_cutlass_api_warp_smoke(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return _run_warp_api_smoke(tmp_path, api_module=CUTLASS_API_MODULE)


def _run_warp_factory_api_smoke(
    tmp_path: Path,
    *,
    api_module: str,
) -> subprocess.CompletedProcess[str]:
    script_name = api_module.replace(".", "_")
    script_path = tmp_path / f"{script_name}_vector_warp_factory_smoke.py"
    write_cutlass_api_vector_warp_factory_smoke(
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


def _run_cutlass_api_warp_factory_smoke(
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_warp_factory_api_smoke(tmp_path, api_module=CUTLASS_API_MODULE)


@pytest.mark.xfail(
    reason=(
        "the group-first planner rejects multi-item warp scans (CUB WarpScan "
        "is scalar-per-lane); the example's ThreadData warp prefixes need a "
        "planner route to the scoped WarpScan wrapper strategy"
    ),
    strict=True,
)
def test_prims_vector_warp_prefix_example_runtime(tmp_path: Path):
    result = _run_prims_example("prims_vector_warp_prefix.py", tmp_path)

    assert result.returncode == 0, result.stderr
    assert "'prefix_out':" in result.stdout
    assert "'valid_prefix_first_warp':" in result.stdout
    assert "'warp_totals':" in result.stdout
    assert "'valid_warp_totals':" in result.stdout
    assert "'warp_min':" in result.stdout
    assert "'warp_max':" in result.stdout
    assert "'valid_warp_max':" in result.stdout
    assert "'warp_xor':" in result.stdout
    assert "'direct_copy':" in result.stdout
    assert "'exchange_out':" in result.stdout


def test_cutlass_api_prims_vector_warp_prefix_runtime(tmp_path: Path):
    result = _run_cutlass_api_warp_smoke(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"primitive_modules":' in result.stdout
    assert result.stdout.count('"cuda.coop.cutlass._block"') == 1
    assert result.stdout.count('"cuda.coop.cutlass._warp"') == 7
    assert '"prefix_out":' in result.stdout
    assert '"valid_prefix_first_warp":' in result.stdout
    assert '"warp_totals":' in result.stdout
    assert '"valid_warp_totals":' in result.stdout
    assert '"warp_min":' in result.stdout
    assert '"warp_max":' in result.stdout
    assert '"valid_warp_max":' in result.stdout
    assert '"warp_xor":' in result.stdout
    assert '"direct_copy":' in result.stdout
    assert '"exchange_out":' in result.stdout


def test_cutlass_api_prims_vector_warp_factory_runtime(tmp_path: Path):
    result = _run_cutlass_api_warp_factory_smoke(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"factory_scopes":' in result.stdout
    assert result.stdout.count('"cuda.coop.cutlass._warp"') == 7
    assert '"prefix_out":' in result.stdout
    assert '"warp_totals":' in result.stdout
    assert '"valid_warp_totals":' in result.stdout
    assert '"warp_min":' in result.stdout
    assert '"warp_max":' in result.stdout
    assert '"valid_warp_max":' in result.stdout
    assert '"warp_xor":' in result.stdout
    assert '"direct_copy":' in result.stdout
    assert '"exchange_out":' in result.stdout
