# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ....support.fixtures.cutlass_prims_runtime_source import (
    write_cutlass_api_load_store_factory_smoke,
    write_cutlass_api_warp_load_store_factory_smoke,
    write_cutlass_prims_array_load_store_smoke,
)
from ..support.prims_runtime import (
    CUTLASS_API_MODULE,
    SOURCE_ROOT,
    _prims_runtime_env,
)


def _run_load_store_factory_smoke(
    tmp_path: Path,
    *,
    api_module: str,
) -> subprocess.CompletedProcess[str]:
    script_name = api_module.replace(".", "_")
    script_path = tmp_path / f"{script_name}_load_store_factory_smoke.py"
    write_cutlass_api_load_store_factory_smoke(
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


def _run_prims_array_load_store_smoke(
    tmp_path: Path,
    *,
    api_module: str,
) -> subprocess.CompletedProcess[str]:
    script_name = api_module.replace(".", "_")
    script_path = tmp_path / f"{script_name}_prims_array_load_store_smoke.py"
    write_cutlass_prims_array_load_store_smoke(
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


def _run_cutlass_api_load_store_factory_smoke(
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_load_store_factory_smoke(
        tmp_path,
        api_module=CUTLASS_API_MODULE,
    )


def _run_warp_load_store_factory_smoke(
    tmp_path: Path,
    *,
    api_module: str,
) -> subprocess.CompletedProcess[str]:
    script_name = api_module.replace(".", "_")
    script_path = tmp_path / f"{script_name}_warp_load_store_factory_smoke.py"
    write_cutlass_api_warp_load_store_factory_smoke(
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


def _run_cutlass_api_warp_load_store_factory_smoke(
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_warp_load_store_factory_smoke(
        tmp_path,
        api_module=CUTLASS_API_MODULE,
    )


def test_cutlass_api_load_store_factory_runtime(tmp_path: Path):
    result = _run_cutlass_api_load_store_factory_smoke(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"factory_scopes":' in result.stdout
    assert result.stdout.count('"cuda.coop.cutlass._block"') == 5
    assert '"direct_copy":' in result.stdout
    assert '"striped_copy":' in result.stdout
    assert '"partial_copy":' in result.stdout
    assert '"valid_store":' in result.stdout
    assert '"exclusive":' in result.stdout


def test_cutlass_api_prims_array_load_store_runtime(tmp_path: Path):
    result = _run_prims_array_load_store_smoke(
        tmp_path,
        api_module=CUTLASS_API_MODULE,
    )

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"load_module": "cuda.coop.cutlass._block"' in result.stdout
    assert '"store_module": "cuda.coop.cutlass._block"' in result.stdout
    assert '"warp_load_module": "cuda.coop.cutlass._warp"' in result.stdout
    assert '"warp_store_module": "cuda.coop.cutlass._warp"' in result.stdout
    assert '"factory_scopes":' in result.stdout
    assert '"array_factory_scopes":' in result.stdout
    assert result.stdout.count('"cuda.coop.cutlass._block"') == 6
    assert result.stdout.count('"cuda.coop.cutlass._warp"') == 6
    assert '"factory_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout
    assert '"warp_factory_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout
    assert '"root_array_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout
    assert '"root_warp_array_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout
    assert '"factory_array_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout
    assert '"warp_factory_array_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout
    assert '"root_implicit_control_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in (result.stdout)
    assert '"root_implicit_control_factory_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in (
        result.stdout
    )
    assert '"root_implicit_control_prefix": [0, 1, 3, 6, 10, 15, 21, 28]' in (
        result.stdout
    )
    assert '"warp_implicit_control_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in (result.stdout)
    assert '"warp_implicit_control_factory_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in (
        result.stdout
    )
    assert '"warp_implicit_control_prefix": [0, 1, 3, 6, 10, 15, 21, 28]' in (
        result.stdout
    )
    assert '"striped_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout
    assert '"metadata_alias_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout
    assert '"root_payload_alias_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout
    assert (
        '"root_payload_factory_alias_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout
    )
    assert '"warp_striped_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout
    assert '"warp_root_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout
    assert '"warp_root_factory_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout
    # Offset 3 selects values 4..94; the final five load items use OOB_DEFAULT.
    # Offset-shifted stores land at [3, 94), leaving five trailing sentinels.
    assert '"partial_tail": [92, 93, 94, -17, -17, -17, -17, -17]' in result.stdout
    assert '"partial_valid_store_tail": [92, 93, 94, -101, -101, -101, -101, -101]' in (
        result.stdout
    )
    assert '"dynamic_offset_copy": [-101, -101, -101, 4, 5, 6, 7, 8]' in result.stdout
    assert '"literal_offset_copy": [1, 2, 3, 4, 5, 6, 7, 8]' in result.stdout


def test_cutlass_api_warp_load_store_factory_runtime(tmp_path: Path):
    result = _run_cutlass_api_warp_load_store_factory_smoke(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"factory_scopes":' in result.stdout
    assert result.stdout.count('"cuda.coop.cutlass._warp"') == 5
    assert '"direct_copy":' in result.stdout
    assert '"striped_copy":' in result.stdout
    assert '"partial_copy":' in result.stdout
    assert '"valid_store":' in result.stdout
    assert '"exclusive":' in result.stdout
