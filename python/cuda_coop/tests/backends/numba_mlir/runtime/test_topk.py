# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Block TopK membership, pairing, and input-preservation checks."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


@pytest.mark.parametrize("selection", ["min", "max"])
@pytest.mark.parametrize("pairs", [False, True])
@pytest.mark.parametrize("qualified", [False, True])
def test_topk_preserves_inputs_and_pairs(selection, pairs, qualified):
    api = numba_coop if qualified else coop
    topk = getattr(api, f"topk_{selection}_{'pairs' if pairs else 'keys'}")

    @cuda.jit
    def kernel(source, selected, associated, preserved, k, count):
        block = api.this_block()
        keys = api.ThreadData(2)
        api.load(block, source, keys)
        if pairs:
            values = api.ThreadData(2, dtype=np.int64)
            values[0] = np.int64(cuda.threadIdx.x * 2)
            values[1] = np.int64(cuda.threadIdx.x * 2 + 1)
            chosen, indices = topk(block, keys, values, k=k, valid_items=count)
            api.store(block, associated, indices, valid_items=min(k, count))
        else:
            chosen = topk(block, keys, k=k, valid_items=count)
        api.store(block, selected, chosen, valid_items=min(k, count))
        api.store(block, preserved, keys)

    source = ((np.arange(128) * 17) % 131 - 64).astype(np.int32)
    selected = np.empty_like(source)
    preserved = np.empty_like(source)
    indices = np.empty(128, dtype=np.int64)
    kernel[1, 64](source, selected, indices, preserved, 17, 103)
    cuda.synchronize()
    expected = (
        np.sort(source[:103])[:17]
        if selection == "min"
        else np.sort(source[:103])[-17:]
    )
    np.testing.assert_array_equal(np.sort(selected[:17]), expected)
    np.testing.assert_array_equal(preserved, source)
    if pairs:
        np.testing.assert_array_equal(selected[:17], source[indices[:17]])
        assert len(set(indices[:17])) == 17


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
    ],
)
@pytest.mark.parametrize("selection", ["min", "max"])
def test_numeric_profile_and_partial_counts(dtype, selection):
    topk = getattr(coop, f"topk_{selection}_keys")

    @cuda.jit
    def kernel(source, destination, k, count):
        keys = coop.ThreadData(2, dtype=dtype)
        coop.load(coop.this_block(), source, keys)
        chosen = topk(coop.this_block(), keys, k=k, valid_items=count)
        coop.store(coop.this_block(), destination, chosen, valid_items=min(k, count))

    source = ((np.arange(128) * 17) % 127).astype(dtype)
    if np.issubdtype(dtype, np.signedinteger) or np.issubdtype(dtype, np.floating):
        source -= dtype(63)
    output = np.empty_like(source)
    for k, count in [(0, 128), (0, 0), (17, 0), (17, 7), (128, 128), (7, 91)]:
        kernel[1, 64](source, output, np.int64(k), np.int64(count))
        cuda.synchronize()
        n = min(k, count)
        ordered = np.sort(source[:count])
        expected = ordered[:n] if selection == "min" else ordered[count - n :]
        np.testing.assert_array_equal(np.sort(output[:n]), expected)


@pytest.mark.parametrize("threads", [1, 16, 32, 128])
@pytest.mark.parametrize("manual_sync", [False, True])
def test_static_controls_and_reused_storage(threads, manual_sync):
    keep = min(3, threads * 2)
    auto_sync = not manual_sync

    @cuda.jit
    def kernel(source, output):
        scratch = coop.TempStorage(auto_sync=auto_sync)
        block = coop.this_block()
        keys = coop.ThreadData(2)
        coop.load(block, source, keys)
        chosen = coop.topk_min_keys(block, keys, k=keep, temp_storage=scratch)
        if manual_sync:
            cuda.syncthreads()
        chosen = coop.topk_max_keys(
            block, chosen, k=1, valid_items=keep, temp_storage=scratch
        )
        if manual_sync:
            cuda.syncthreads()
        coop.store(block, output, chosen, valid_items=1)

    source = np.arange(threads * 2, 0, -1, dtype=np.int32)
    output = np.zeros_like(source)
    kernel[1, threads](source, output)
    cuda.synchronize()
    assert output[0] == keep


def test_ties_preserve_selected_pairs_and_float_zero_bits():
    @cuda.jit
    def kernel(source, output, indices, preserved):
        keys = cuda.local.array(2, dtype=np.float32)
        values = cuda.local.array(2, dtype=np.int32)
        for i in range(2):
            values[i] = cuda.threadIdx.x * 2 + i
            keys[i] = source[values[i]]
        chosen, positions = numba_coop.topk_min_pairs(
            numba_coop.this_block(), keys, values, k=17
        )
        numba_coop.store(numba_coop.this_block(), output, chosen, valid_items=17)
        numba_coop.store(numba_coop.this_block(), indices, positions, valid_items=17)
        numba_coop.store(numba_coop.this_block(), preserved, keys)

    source = np.tile(np.array([-0.0, 0.0, 1.0, 2.0], dtype=np.float32), 32)
    output = np.empty_like(source)
    preserved = np.empty_like(source)
    indices = np.empty(128, dtype=np.int32)
    kernel[1, 64](source, output, indices, preserved)
    cuda.synchronize()
    assert np.all(output[:17] == 0)
    assert len(set(indices[:17])) == 17
    np.testing.assert_array_equal(
        output[:17].view(np.uint32), source[indices[:17]].view(np.uint32)
    )
    np.testing.assert_array_equal(preserved.view(np.uint32), source.view(np.uint32))


@pytest.mark.parametrize(
    "k,count", [(-1, 128), (129, 128), (2**32, 128), (7, -1), (7, 129), (7, 2**32)]
)
def test_invalid_runtime_counts_trap_before_narrowing(k, count):
    # A trap poisons the CUDA context, so each invalid launch needs a child.
    script = f"""
import numpy as np
from pathlib import Path
from numba_cuda_mlir import cuda
from cuda import coop
assert Path(coop.__file__).resolve() == Path({str(Path(coop.__file__).resolve())!r})
@cuda.jit
def kernel(source, output, k, count):
    keys = coop.ThreadData(2)
    coop.load(coop.this_block(),source,keys)
    chosen = coop.topk_min_keys(coop.this_block(),keys,k=k,valid_items=count)
    coop.store(coop.this_block(),output,chosen,valid_items=1)
source = cuda.to_device(np.arange(128,dtype=np.int32))
output = cuda.device_array(128,dtype=np.int32)
try:
    kernel[1,64](source,output,np.int64({k}),np.int64({count}))
    cuda.synchronize()
except Exception as error:
    message = str(error)
    assert any(token in message for token in ("CUDA_ERROR_ILLEGAL_INSTRUCTION", "CUDA_ERROR_LAUNCH_FAILED", "illegal instruction", "unspecified launch failure")), message
    print("EXPECTED_TOPK_TRAP")
else:
    raise AssertionError("invalid TopK control did not trap")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXPECTED_TOPK_TRAP" in result.stdout


def test_empty_prefix_with_cub_assertions(monkeypatch):
    from cuda.coop.numba_mlir._compiler import _nvrtc

    compile_provider = _nvrtc.compile

    def with_assertions(*args, **kwargs):
        if "BlockTopKCoop" in kwargs.get("cpp", ""):
            kwargs["cpp"] = "#define CCCL_ENABLE_ASSERTIONS\n" + kwargs["cpp"]
        return compile_provider(*args, **kwargs)

    monkeypatch.setattr(_nvrtc, "compile", with_assertions)

    @cuda.jit
    def kernel(source, output, k, count):
        keys = coop.ThreadData(3)
        block = coop.this_block()
        coop.load(block, source, keys)
        selected = coop.topk_min_keys(block, keys, k=k, valid_items=count)
        coop.store(block, output, selected, valid_items=min(k, count))

    source = np.arange(96, dtype=np.int32)
    output = np.full_like(source, -1)
    for k, count in ((0, 96), (0, 0), (17, 0)):
        kernel[1, 32](source, output, np.int64(k), np.int64(count))
        cuda.synchronize()
        np.testing.assert_array_equal(output, -np.ones_like(source))


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("pairs", [False, True])
@pytest.mark.parametrize("threads,items", [(3, 5), (37, 3), (65, 1)])
def test_chained_results_infer_dtype_from_indexed_writes(
    qualified, pairs, threads, items
):
    api = numba_coop if qualified else coop
    keep = threads * items // 2
    selected_count = min(3, keep)

    @cuda.jit
    def kernel(source, positions, output, indices, preserved):
        block = api.this_block()
        keys = api.ThreadData(items)
        values = api.ThreadData(items)
        for item in range(items):
            index = cuda.threadIdx.x * items + item
            keys[item] = source[index]
            values[item] = positions[index]
        if pairs:
            first_keys, first_values = api.topk_min_pairs(block, keys, values, k=keep)
            chosen, associated = api.topk_max_pairs(
                block, first_keys, first_values, k=selected_count, valid_items=keep
            )
            api.store(block, indices, associated, valid_items=selected_count)
        else:
            first = api.topk_min_keys(block, keys, k=keep)
            chosen = api.topk_max_keys(block, first, k=selected_count, valid_items=keep)
        api.store(block, output, chosen, valid_items=selected_count)
        api.store(block, preserved, keys)

    source = ((np.arange(threads * items) * 17 + 11) % 97 - 48).astype(np.int16)
    positions = np.arange(source.size, dtype=np.int64)
    output = np.zeros_like(source)
    indices = np.zeros_like(positions)
    preserved = np.zeros_like(source)
    kernel[1, threads](source, positions, output, indices, preserved)
    cuda.synchronize()
    np.testing.assert_array_equal(
        np.sort(output[:selected_count]), np.sort(source)[keep - selected_count : keep]
    )
    np.testing.assert_array_equal(preserved, source)
    if pairs:
        assert len(set(indices[:selected_count])) == selected_count
        np.testing.assert_array_equal(
            output[:selected_count], source[indices[:selected_count]]
        )
