# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Block neighbor kernels checked against independent host oracles."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)
from numba_cuda_mlir import types

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

_DTYPES = [
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
]


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("direction", ["left", "right"])
def test_adjacent_partial_boundaries_and_input_preservation(dtype, direction):
    source = ((np.arange(90) * 7) % 17).astype(dtype)
    compiler_dtype = getattr(types, source.dtype.name)

    @cuda.jit
    def kernel(source, output, original, count):
        block = coop.this_block()
        values = coop.ThreadData(3)
        thread = cuda.threadIdx.x + 5 * (cuda.threadIdx.y + 3 * cuda.threadIdx.z)
        for i in range(3):
            values[i] = source[thread * 3 + i]
        if direction == "left":
            result = coop.adjacent_difference(
                block,
                values,
                valid_items=count,
                tile_predecessor_item=compiler_dtype(2),
                temp_storage=coop.TempStorage(),
            )
        else:
            result = coop.adjacent_difference(
                block, values, valid_items=count, direction="right"
            )
        for i in range(3):
            output[thread * 3 + i] = result[i]
            original[thread * 3 + i] = values[i]

    for count in (0, 1, 2, 3, 4, 29, 89, 90):
        expected = source.copy()
        if direction == "left":
            if count:
                expected[0] = np.subtract(source[0], dtype(2), dtype=dtype)
                expected[1:count] = source[1:count] - source[: count - 1]
        elif count:
            expected[: count - 1] = source[: count - 1] - source[1:count]
        output = np.empty_like(source)
        original = np.empty_like(source)
        kernel[1, (5, 3, 2)](source, output, original, np.int64(count))
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(original, source)


@pytest.mark.parametrize("mode", ["heads", "tails", "heads_and_tails"])
@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("dtype", [np.int16, np.uint64, np.float32, np.float64])
def test_flags_boundaries_pair_results_and_chained_scan(dtype, mode, boundary):
    source = (np.arange(96) // 5 % 7).astype(dtype)
    source[31:35] = 9
    # Both modes treat comparison operands in increasing input-index order.
    compiler_dtype = getattr(types, source.dtype.name)

    @cuda.jit
    def kernel(source, head_out, tail_out, prefix_out, original):
        block = coop.this_block()
        values = coop.ThreadData(3)
        coop.load(block, source, values)
        if mode == "heads_and_tails":
            if boundary:
                heads, tails = coop.discontinuity(
                    block,
                    values,
                    mode=mode,
                    tile_predecessor_item=compiler_dtype(0),
                    tile_successor_item=compiler_dtype(5),
                )
            else:
                heads, tails = coop.discontinuity(block, values, mode=mode)
            coop.store(block, head_out, heads)
            coop.store(block, tail_out, tails)
            scanned = coop.inclusive_sum(block, tails)
        elif mode == "heads":
            if boundary:
                result = coop.discontinuity(
                    block, values, mode=mode, tile_predecessor_item=compiler_dtype(0)
                )
            else:
                result = coop.discontinuity(block, values, mode=mode)
            coop.store(block, head_out, result)
            scanned = coop.inclusive_sum(block, result)
        else:
            if boundary:
                result = coop.discontinuity(
                    block, values, mode=mode, tile_successor_item=compiler_dtype(5)
                )
            else:
                result = coop.discontinuity(block, values, mode=mode)
            coop.store(block, tail_out, result)
            scanned = coop.inclusive_sum(block, result)
        coop.store(block, prefix_out, scanned)
        coop.store(block, original, values)

    heads = np.r_[
        int(source[0] != 0) if boundary else 1, source[1:] != source[:-1]
    ].astype(np.int32)
    tails = np.r_[
        source[:-1] != source[1:], int(source[-1] != 5) if boundary else 1
    ].astype(np.int32)
    head_out, tail_out, prefix_out = [np.empty(96, dtype=np.int32) for _ in range(3)]
    original = np.empty_like(source)
    kernel[1, 32](source, head_out, tail_out, prefix_out, original)
    if mode != "tails":
        np.testing.assert_array_equal(head_out, heads)
    if mode != "heads":
        np.testing.assert_array_equal(tail_out, tails)
    np.testing.assert_array_equal(
        prefix_out, np.cumsum(heads if mode == "heads" else tails)
    )
    np.testing.assert_array_equal(original, source)


def test_qualified_custom_operators_and_shared_storage_reuse():
    def difference(current, neighbor):
        return current * 2 - neighbor

    def flag(previous, current):
        return previous < current

    @cuda.jit
    def kernel(source, output, head_out, tail_out, original):
        block = numba_coop.this_block()
        scratch = numba_coop.TempStorage()
        values = cuda.local.array(3, dtype=types.int32)
        for i in range(3):
            values[i] = source[cuda.threadIdx.x * 3 + i]
        differences = numba_coop.adjacent_difference(
            block,
            values,
            direction="right",
            tile_successor_item=7,
            difference_op=difference,
            temp_storage=scratch,
        )
        heads, tails = numba_coop.discontinuity(
            block,
            differences,
            mode="heads_and_tails",
            flag_op=flag,
            tile_predecessor_item=4,
            tile_successor_item=8,
            temp_storage=scratch,
        )
        numba_coop.store(block, output, differences)
        numba_coop.store(block, head_out, heads)
        numba_coop.store(block, tail_out, tails)
        for i in range(3):
            original[cuda.threadIdx.x * 3 + i] = values[i]

    source = ((np.arange(96) * 7) % 19).astype(np.int32)
    expected = source * 2 - np.r_[source[1:], np.int32(7)]
    outputs = [np.empty(96, dtype=np.int32) for _ in range(4)]
    kernel[1, 32](source, *outputs)
    np.testing.assert_array_equal(outputs[0], expected)
    np.testing.assert_array_equal(
        outputs[1], np.r_[4 < expected[0], expected[:-1] < expected[1:]]
    )
    np.testing.assert_array_equal(
        outputs[2], np.r_[expected[:-1] < expected[1:], expected[-1] < 8]
    )
    np.testing.assert_array_equal(outputs[3], source)


def test_multiblock_delta_example():
    # adjacent-difference-example-begin
    @cuda.jit
    def delta_encode(source, output):
        # Use distinct source/output arrays: another block needs the preceding
        # source tile's last value, so an in-place launch would race.
        block = coop.this_block()
        base = cuda.blockIdx.x * 512
        values = coop.ThreadData(4)
        coop.load(block, source, values, offset=base)
        previous = np.int32(0)
        if base > 0:
            previous = source[base - 1]
        differences = coop.adjacent_difference(
            block, values, direction="left", tile_predecessor_item=previous
        )
        coop.store(block, output, differences, offset=base)

    # adjacent-difference-example-end

    source = ((np.arange(1536) * 13) % 101).astype(np.int32)
    output = np.empty_like(source)
    delta_encode[3, 128](source, output)
    np.testing.assert_array_equal(output, np.diff(source, prepend=np.int32(0)))


def test_tile_run_labels_example():
    # discontinuity-example-begin
    @cuda.jit
    def label_runs_per_tile(keys, output):
        # Full 512-item tiles; IDs restart at zero in each block.
        block = coop.this_block()
        base = cuda.blockIdx.x * 512
        values = coop.ThreadData(4)
        coop.load(block, keys, values, offset=base)
        heads = coop.discontinuity(block, values, mode="heads")
        labels = coop.inclusive_sum(block, heads)
        for i in range(4):
            labels[i] -= 1
        coop.store(block, output, labels, offset=base)

    # discontinuity-example-end

    source = (np.arange(1024) // 13).astype(np.int32)
    output = np.empty_like(source)
    label_runs_per_tile[2, 128](source, output)
    for base in (0, 512):
        tile = source[base : base + 512]
        expected = np.cumsum(np.r_[1, tile[1:] != tile[:-1]]) - 1
        np.testing.assert_array_equal(output[base : base + 512], expected)


@pytest.mark.parametrize("direction", ["left", "right"])
def test_full_tile_default_boundary_and_chained_result(direction):
    @cuda.jit
    def kernel(source, output):
        block = coop.this_block()
        values = coop.ThreadData(1)
        values[0] = source[cuda.threadIdx.x]
        first = coop.adjacent_difference(block, values, direction=direction)
        second = numba_coop.adjacent_difference(block, first, direction=direction)
        coop.store(block, output, second)

    source = ((np.arange(65) * 13) % 29).astype(np.int32)
    expected = source.copy()
    for _ in range(2):
        original = expected.copy()
        if direction == "left":
            expected[1:] = original[1:] - original[:-1]
        else:
            expected[:-1] = original[:-1] - original[1:]
    output = np.empty_like(source)
    kernel[1, 65](source, output)
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize("count", [-1, 193, 1 << 32])
def test_invalid_runtime_count_traps_before_narrowing(count):
    import subprocess
    import sys
    from pathlib import Path

    script = f"""
import numpy as np
from numba_cuda_mlir import cuda
from cuda import coop
from pathlib import Path
assert Path(coop.__file__).resolve() == Path({str(Path(coop.__file__).resolve())!r})
@cuda.jit
def kernel(source, output, count):
    values = coop.ThreadData(3)
    coop.load(coop.this_block(), source, values)
    result = coop.adjacent_difference(coop.this_block(), values, valid_items=count)
    coop.store(coop.this_block(), output, result)
source = np.arange(192, dtype=np.int32)
output = np.empty_like(source)
kernel[1, 64](source, output, np.int64({count}))
cuda.synchronize()
raise AssertionError('invalid count did not trap')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-P" if sys.version_info >= (3, 11) else "-I",
            "-B",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert any(
        error in output
        for error in ("CUDA_ERROR_ILLEGAL_INSTRUCTION", "CUDA_ERROR_LAUNCH_FAILED")
    ), output
