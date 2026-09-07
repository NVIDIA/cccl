# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Cold-activation and final-link qualification for common-root Numba TopK."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

from ....support.paths import TESTS_ROOT

pytestmark = [pytest.mark.gpu, pytest.mark.link, pytest.mark.qualification]

_THREADS = 32
_ITEMS_PER_THREAD = 2
_K = 7
_COLD_ACTIVATION_PROBE = (
    TESTS_ROOT / "support" / "fixtures" / "numba_root_topk_cold_activation.py"
)


@cuda.jit
def _root_topk_max_final_link_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        keys[item] = d_input[index]

    selected = coop.topk_max_keys(coop.this_block(), keys, _K)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        d_output[index] = selected[item]


@cuda.jit
def _root_topk_min_final_link_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        keys[item] = d_input[index]

    selected = coop.topk_min_keys(coop.this_block(), keys, _K)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        d_output[index] = selected[item]


@cuda.jit
def _qualified_topk_max_final_link_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    keys = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        keys[item] = d_input[index]

    selected = numba_coop.topk_max_keys(numba_coop.this_block(), keys, _K)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        d_output[index] = selected[item]


@cuda.jit
def _qualified_topk_min_final_link_kernel(d_input, d_output):
    tid = cuda.threadIdx.x
    keys = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        keys[item] = d_input[index]

    selected = numba_coop.topk_min_keys(numba_coop.this_block(), keys, _K)
    for item in range(_ITEMS_PER_THREAD):
        index = tid * _ITEMS_PER_THREAD + item
        d_output[index] = selected[item]


def _assert_provider_eliminated(kernel) -> None:
    sass = kernel.inspect_sass(kernel.signatures[0])
    assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None
    assert (
        re.search(
            r"(?im)^\s*(?:Function\s*:|\.section\s+\.text\.)[^\n]*"
            r"cuda_coop_numba_mlir_block_topk",
            sass,
        )
        is None
    )


def test_root_topk_activates_numba_automatically_in_fresh_process(
    tmp_path: Path,
):
    environment = os.environ.copy()
    environment.pop("CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION", None)
    environment["NUMBA_CACHE_DIR"] = str(tmp_path / "numba-cache")
    result = subprocess.run(
        [sys.executable, str(_COLD_ACTIVATION_PROBE)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        "Numba root-TopK cold-activation subprocess failed\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.evidence_for(
    "group.topk_max_keys",
    backend="numba_mlir",
    evidence="link",
)
@pytest.mark.evidence_for(
    "group.topk_min_keys",
    backend="numba_mlir",
    evidence="link",
)
def test_common_and_qualified_topk_execute_and_eliminate_provider_calls():
    item_count = _THREADS * _ITEMS_PER_THREAD
    values = ((np.arange(item_count, dtype=np.int32) * 29) % item_count).astype(
        np.int32
    )
    if shutil.which("nvdisasm") is None:
        pytest.skip("requires nvdisasm to inspect the final cubin")

    cases = (
        (_root_topk_max_final_link_kernel, np.sort(values)[-_K:]),
        (_qualified_topk_max_final_link_kernel, np.sort(values)[-_K:]),
        (_root_topk_min_final_link_kernel, np.sort(values)[:_K]),
        (_qualified_topk_min_final_link_kernel, np.sort(values)[:_K]),
    )
    for kernel, expected in cases:
        output = np.zeros_like(values)
        kernel[1, _THREADS](values, output)
        np.testing.assert_array_equal(np.sort(output[:_K]), expected)
        _assert_provider_eliminated(kernel)


@pytest.mark.evidence_for("group.topk_max_pairs", backend="numba_mlir", evidence="link")
@pytest.mark.evidence_for("group.topk_min_pairs", backend="numba_mlir", evidence="link")
def test_common_and_qualified_topk_pair_providers_are_eliminated(
    backend_prerequisite,
):
    if shutil.which("nvdisasm") is None:
        pytest.skip("requires nvdisasm to inspect the final cubin")
    backend_prerequisite(
        "numba_mlir",
        cuda.is_available(),
        "CUDA GPU required for Numba-CUDA-MLIR tests",
    )
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    from ....backends.numba_mlir.support.compile import compile_for_launch

    @cuda.jit
    def kernel(key_source, value_source, key_output, value_output, k):
        tid = cuda.threadIdx.x
        common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        common_values = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.float32)
        qualified_keys = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        qualified_values = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.float32)
        for index in range(_ITEMS_PER_THREAD):
            offset = tid * _ITEMS_PER_THREAD + index
            common_keys[index] = key_source[offset]
            common_values[index] = value_source[offset]
            qualified_keys[index] = key_source[offset]
            qualified_values[index] = value_source[offset]
        common_max_keys, common_max_values = coop.topk_max_pairs(
            coop.this_block(), common_keys, common_values, k
        )
        qualified_max_keys, qualified_max_values = numba_coop.topk_max_pairs(
            numba_coop.this_block(), qualified_keys, qualified_values, k
        )
        common_min_keys, common_min_values = coop.topk_min_pairs(
            coop.this_block(), common_keys, common_values, k, valid_items=51
        )
        qualified_min_keys, qualified_min_values = numba_coop.topk_min_pairs(
            numba_coop.this_block(),
            qualified_keys,
            qualified_values,
            k,
            valid_items=51,
        )
        key_output[tid] = (
            common_max_keys[0]
            + qualified_max_keys[0]
            + common_min_keys[0]
            + qualified_min_keys[0]
        )
        value_output[tid] = (
            common_max_values[0]
            + qualified_max_values[0]
            + common_min_values[0]
            + qualified_min_values[0]
        )

    key_array = types.Array(types.int32, 1, "C")
    value_array = types.Array(types.float32, 1, "C")
    signature = cuda_typing.signature(
        types.none,
        key_array,
        value_array,
        key_array,
        value_array,
        types.int32,
    )
    inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    pair_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockTopKCoop"
    ]
    assert {record[1] for record in pair_records} == {
        "max_pairs_full",
        "min_pairs_partial",
    }
    symbols = {record[2] for record in pair_records}
    sass = kernel.inspect_sass(inspect_key)
    assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None
    for symbol in symbols:
        assert (
            re.search(
                rf"(?im)^\s*(?:Function\s*:|\.section\s+\.text\.)"
                rf"[^\n]*{re.escape(symbol)}",
                sass,
            )
            is None
        )
