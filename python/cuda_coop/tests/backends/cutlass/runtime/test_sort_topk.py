# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ....support.fixtures.cutlass_prims_runtime_source import (
    write_cutlass_mixed_payload_factory_smoke,
    write_prims_api_vector_sort_topk_smoke,
)
from ..support.prims_runtime import (
    CUTLASS_API_MODULE,
    SOURCE_ROOT,
    _prims_runtime_env,
    _run_prims_example,
)


def _run_mixed_payload_factory_smoke(
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    script_path = tmp_path / "cuda_coop_cutlass_mixed_payload_factory_smoke.py"
    write_cutlass_mixed_payload_factory_smoke(
        script_path,
        source_root=SOURCE_ROOT,
    )
    result = subprocess.run(
        [sys.executable, "-B", str(script_path)],
        check=False,
        capture_output=True,
        env=_prims_runtime_env(tmp_path),
        text=True,
    )

    return result


def _run_block_api_smoke(
    tmp_path: Path,
    *,
    api_module: str,
    call_mode: str,
) -> subprocess.CompletedProcess[str]:
    script_name = api_module.replace(".", "_")
    script_path = tmp_path / f"{script_name}_vector_sort_topk_{call_mode}_smoke.py"
    write_prims_api_vector_sort_topk_smoke(
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


def _run_cutlass_api_factory_smoke(
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_block_api_smoke(
        tmp_path,
        api_module=CUTLASS_API_MODULE,
        call_mode="factory",
    )


def _run_cutlass_api_direct_smoke(
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    return _run_block_api_smoke(
        tmp_path,
        api_module=CUTLASS_API_MODULE,
        call_mode="direct",
    )


def test_prims_vector_sort_topk_example_runtime(tmp_path: Path):
    result = _run_prims_example("prims_vector_sort_topk.py", tmp_path)

    assert result.returncode == 0, result.stderr
    assert "'topk_valid_items':" in result.stdout
    assert "'sorted_keys':" in result.stdout
    assert "'top_keys':" in result.stdout


def test_prims_vector_pair_sort_topk_example_runtime(tmp_path: Path):
    result = _run_prims_example("prims_vector_pair_sort_topk.py", tmp_path)

    assert result.returncode == 0, result.stderr
    assert "'topk_valid_items':" in result.stdout
    assert "'sorted_pairs':" in result.stdout
    assert "'top_pairs':" in result.stdout


def test_mixed_payload_sort_topk_example_runtime(tmp_path: Path):
    result = _run_prims_example("mixed_payload_sort_topk.py", tmp_path)

    assert result.returncode == 0, result.stderr
    assert "'sorted_vector_keys':" in result.stdout
    assert "'top_vector_keys':" in result.stdout
    assert "'sorted_fragment_keys':" in result.stdout


def test_mixed_payload_factory_sort_topk_example_runtime(tmp_path: Path):
    result = _run_prims_example("mixed_payload_factory_sort_topk.py", tmp_path)

    assert result.returncode == 0, result.stderr
    assert "'factory_scopes':" in result.stdout
    assert "'sorted_vector_keys':" in result.stdout
    assert "'top_vector_keys':" in result.stdout
    assert "'sorted_tensor_keys':" in result.stdout


def test_cutlass_api_mixed_payload_factory_runtime(tmp_path: Path):
    result = _run_mixed_payload_factory_smoke(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"factory_scopes":' in result.stdout
    assert result.stdout.count('"cuda.coop.cutlass._block"') == 5
    assert '"topk_valid_items":' in result.stdout
    assert '"sorted_vector_keys":' in result.stdout
    assert '"top_vector_keys":' in result.stdout
    assert '"sorted_fragment_keys":' in result.stdout


def test_cutlass_api_prims_vector_sort_topk_factory_runtime(tmp_path: Path):
    result = _run_cutlass_api_factory_smoke(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"factory_scopes":' in result.stdout
    assert result.stdout.count('"cuda.coop.cutlass._block"') == 4
    assert '"topk_valid_items":' in result.stdout
    assert '"sorted_keys":' in result.stdout
    assert '"top_keys":' in result.stdout
    assert '"sorted_pairs":' in result.stdout
    assert '"top_pairs":' in result.stdout


def test_cutlass_api_prims_vector_sort_topk_direct_runtime(tmp_path: Path):
    result = _run_cutlass_api_direct_smoke(tmp_path)

    assert result.returncode == 0, result.stderr
    assert '"api_module": "cuda.coop.cutlass"' in result.stdout
    assert '"primitive_modules":' in result.stdout
    assert result.stdout.count('"cuda.coop.cutlass._block"') == 5
    assert '"topk_valid_items":' in result.stdout
    assert '"sorted_keys":' in result.stdout
    assert '"top_keys":' in result.stdout
    assert '"sorted_pairs":' in result.stdout
    assert '"top_pairs":' in result.stdout
