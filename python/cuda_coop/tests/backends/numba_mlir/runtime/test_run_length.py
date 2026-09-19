# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""RLD window masking, prepared-state reuse and checked boundary behavior."""

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


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize(
    "value_dtype,length_dtype",
    [(np.int8, np.uint16), (np.float32, np.int64), (np.float64, np.uint64)],
)
def test_windows_preserve_inputs_and_zero_invalid_slots(
    qualified, value_dtype, length_dtype
):
    api = numba_coop if qualified else coop

    @cuda.jit
    def kernel(values, lengths, output, kept_values, kept_lengths, offset):
        block = api.this_block()
        run_values = api.ThreadData(2)
        run_lengths = api.ThreadData(2)
        api.load(block, values, run_values)
        api.load(block, lengths, run_lengths)
        decoded = api.run_length_decode(
            block,
            run_values,
            run_lengths,
            decoded_items_per_thread=3,
            decoded_window_offset=offset,
        )
        api.store(block, output, decoded)
        api.store(block, kept_values, run_values)
        api.store(block, kept_lengths, run_lengths)

    values = np.arange(64, dtype=value_dtype) + 7
    lengths = np.zeros(64, dtype=length_dtype)
    output = np.full(96, 99, dtype=value_dtype)
    kept_values = np.zeros_like(values)
    kept_lengths = np.zeros_like(lengths)
    for prefix in ([], [3, 2], [1] * 64, [10, 200, 4]):
        lengths[:] = 0
        lengths[: len(prefix)] = prefix
        decoded = np.repeat(values, lengths.astype(np.int64))
        for offset in (0, 2, len(decoded), len(decoded) + 1, 2**32 + 2):
            kernel[1, 32](
                values, lengths, output, kept_values, kept_lengths, np.uint64(offset)
            )
            cuda.synchronize()
            expected = np.zeros_like(output)
            window = decoded[offset : offset + 96]
            expected[: len(window)] = window
            np.testing.assert_array_equal(output, expected)
            np.testing.assert_array_equal(kept_values, values)
            np.testing.assert_array_equal(kept_lengths, lengths)


@pytest.mark.parametrize("auto_sync", [True, False])
@pytest.mark.parametrize("offset_dtype", [np.uint32, np.uint64])
def test_auxiliary_outputs_and_explicit_storage_reuse(auto_sync, offset_dtype):
    @cuda.jit
    def kernel(values, lengths, output, relative_output, totals):
        block = numba_coop.this_block()
        scratch = numba_coop.TempStorage(auto_sync=auto_sync)
        runs = cuda.local.array(2, dtype=np.int32)
        sizes = cuda.local.array(2, dtype=np.uint32)
        for i in range(2):
            runs[i] = values[cuda.threadIdx.x * 2 + i]
            sizes[i] = lengths[cuda.threadIdx.x * 2 + i]
        relative = numba_coop.ThreadData(4, dtype=offset_dtype)
        total = numba_coop.ThreadData(1, dtype=offset_dtype)
        first = numba_coop.run_length_decode(
            block,
            runs,
            sizes,
            decoded_items_per_thread=4,
            decoded_window_offset=0,
            total_decoded_size=total,
            relative_offsets=relative,
            decoded_offset_dtype=offset_dtype,
            temp_storage=scratch,
        )
        numba_coop.store(block, output, first)
        if not auto_sync:
            block.sync()
        second = numba_coop.run_length_decode(
            block,
            runs,
            sizes,
            decoded_items_per_thread=4,
            decoded_window_offset=2,
            total_decoded_size=total,
            relative_offsets=relative,
            decoded_offset_dtype=offset_dtype,
            temp_storage=scratch,
        )
        numba_coop.store(block, output, second, offset=128)
        numba_coop.store(block, relative_output, relative)
        totals[cuda.threadIdx.x] = total[0]

    values = np.zeros(64, dtype=np.int32)
    values[:2] = [7, 9]
    lengths = np.zeros(64, dtype=np.uint32)
    lengths[:2] = [3, 2]
    output = np.empty(256, dtype=np.int32)
    relative = np.empty(128, dtype=offset_dtype)
    totals = np.empty(32, dtype=offset_dtype)
    kernel[1, 32](values, lengths, output, relative, totals)
    cuda.synchronize()
    np.testing.assert_array_equal(output[:5], [7, 7, 7, 9, 9])
    np.testing.assert_array_equal(output[128:131], [7, 9, 9])
    assert not output[5:128].any() and not output[131:].any()
    np.testing.assert_array_equal(relative[:3], [2, 0, 1])
    assert np.all(relative[3:] == np.iinfo(offset_dtype).max)
    assert np.all(totals == 5)


@pytest.mark.parametrize("total_size", [2**32 + 5, 2**64 - 65])
def test_uint64_decode_above_uint32_range_without_large_allocation(total_size):
    @cuda.jit
    def kernel(values, lengths, output, relative_output, totals, offset):
        block = numba_coop.this_block()
        runs = numba_coop.ThreadData(1)
        sizes = numba_coop.ThreadData(1)
        numba_coop.load(block, values, runs)
        numba_coop.load(block, lengths, sizes)
        relative = numba_coop.ThreadData(2, dtype=np.uint64)
        total = numba_coop.ThreadData(1, dtype=np.uint64)
        decoded = numba_coop.run_length_decode(
            block,
            runs,
            sizes,
            decoded_items_per_thread=2,
            decoded_window_offset=offset,
            decoded_offset_dtype=np.uint64,
            relative_offsets=relative,
            total_decoded_size=total,
        )
        numba_coop.store(block, output, decoded)
        numba_coop.store(block, relative_output, relative)
        totals[cuda.threadIdx.x] = total[0]

    values = np.zeros(32, dtype=np.int32)
    values[:2] = [7, 9]
    lengths = np.zeros(32, dtype=np.uint64)
    lengths[:2] = [total_size - 2, 2]
    output = np.empty(64, dtype=np.int32)
    relative = np.empty(64, dtype=np.uint64)
    total = np.empty(32, dtype=np.uint64)
    kernel[1, 32](values, lengths, output, relative, total, np.uint64(total_size - 3))
    cuda.synchronize()
    np.testing.assert_array_equal(output[:3], [7, 9, 9])
    np.testing.assert_array_equal(
        relative[:3], np.array([total_size - 3, 0, 1], dtype=np.uint64)
    )
    assert not output[3:].any()
    assert np.all(relative[3:] == np.iinfo(np.uint64).max)
    assert np.all(total == total_size)


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("threads", [32, 37])
def test_bulk_multiple_windows_partial_final_window_and_empty(qualified, threads):
    api = numba_coop if qualified else coop

    @cuda.jit
    def kernel(values, lengths, output, relative, totals, start):
        block = api.this_block()
        scratch = api.TempStorage()
        runs = api.ThreadData(2)
        sizes = api.ThreadData(2)
        api.load(block, values, runs)
        api.load(block, lengths, sizes)
        if qualified:
            total = api.run_length_decode_into(
                block,
                runs,
                sizes,
                output,
                decoded_items_per_thread=3,
                destination_offset=start,
                decoded_offset_dtype=np.uint64,
                relative_offsets=relative,
                temp_storage=scratch,
            )
        else:
            total = api.run_length_decode_into(
                block,
                runs,
                sizes,
                output,
                decoded_items_per_thread=3,
                destination_offset=start,
                temp_storage=scratch,
            )
        totals[cuda.threadIdx.x] = total
        # Reusing this descriptor after the internal bulk loop needs the normal barrier.
        again = api.run_length_decode(
            block, runs, sizes, decoded_items_per_thread=1, temp_storage=scratch
        )
        if cuda.threadIdx.x == 0 and total != 0:
            output[start] = again[0]

    values = np.arange(threads * 2, dtype=np.int16) + 7
    lengths = np.zeros(threads * 2, dtype=np.uint32)
    for prefix in ([], [3, 2], [4] * (threads * 2), [1, 401, 3]):
        lengths[:] = 0
        lengths[: len(prefix)] = prefix
        expected = np.repeat(values, lengths)
        output = np.full(len(expected) + 10, -1, dtype=np.int16)
        relative = np.full(len(output), 9999, dtype=np.uint64)
        totals = np.empty(threads, dtype=np.uint64)
        kernel[1, threads](values, lengths, output, relative, totals, np.int64(3))
        cuda.synchronize()
        np.testing.assert_array_equal(output[3 : 3 + len(expected)], expected)
        assert np.all(output[:3] == -1) and np.all(output[3 + len(expected) :] == -1)
        assert np.all(totals == len(expected))
        if qualified:
            offsets = (
                np.concatenate([np.arange(size, dtype=np.uint64) for size in prefix])
                if prefix
                else []
            )
            np.testing.assert_array_equal(relative[3 : 3 + len(expected)], offsets)
            assert np.all(relative[:3] == 9999) and np.all(
                relative[3 + len(expected) :] == 9999
            )


@pytest.mark.parametrize(
    "case",
    [
        "negative_length",
        "interior_zero",
        "sum_overflow32",
        "value_overflow32",
        "sum_overflow64",
        "negative_offset",
        "capacity",
        "relative_capacity",
    ],
)
def test_invalid_inputs_trap_before_decode_or_bulk_writes(case):
    # A device trap poisons the context; isolate each deliberate invalid launch.
    script = f"""
import numpy as np
from pathlib import Path
from numba_cuda_mlir import cuda
import cuda.coop.numba_mlir as numba_coop
assert Path(numba_coop.__file__).resolve() == Path({str(Path(numba_coop.__file__).resolve())!r})
case = {case!r}
offset_dtype = np.uint64 if case == 'sum_overflow64' else np.uint32
length_dtype = np.int64 if case == 'negative_length' else np.uint64
@cuda.jit
def kernel(values, lengths, output, relative, offset):
    block = numba_coop.this_block()
    runs = numba_coop.ThreadData(1)
    sizes = numba_coop.ThreadData(1)
    numba_coop.load(block, values, runs)
    numba_coop.load(block, lengths, sizes)
    if case == 'capacity' or case == 'relative_capacity':
        numba_coop.run_length_decode_into(block, runs, sizes, output,
            decoded_items_per_thread=2, relative_offsets=relative,
            decoded_offset_dtype=offset_dtype, destination_offset=offset)
    else:
        decoded = numba_coop.run_length_decode(block, runs, sizes,
            decoded_items_per_thread=2, decoded_window_offset=offset,
            decoded_offset_dtype=offset_dtype)
        numba_coop.store(block, output, decoded)
values = np.arange(32, dtype=np.int32)
lengths = np.zeros(32, dtype=length_dtype)
lengths[:2] = [3,2]
if case == 'negative_length': lengths[0] = -1
if case == 'interior_zero': lengths[:3] = [3,0,2]
if case == 'sum_overflow32': lengths[:2] = [2**32-1,1]
if case == 'value_overflow32': lengths[:2] = [2**32,1]
if case == 'sum_overflow64': lengths[:2] = [2**63,2**63]
output = cuda.to_device(np.full(4 if case == 'capacity' else 64,-1,dtype=np.int32))
relative = cuda.device_array(4 if case == 'relative_capacity' else 64,dtype=offset_dtype)
try:
    kernel[1,32](cuda.to_device(values),cuda.to_device(lengths),output,relative,np.int64(-1 if case == 'negative_offset' else 0))
    cuda.synchronize()
except Exception as error:
    message = str(error)
    assert any(token in message for token in ('CUDA_ERROR_ILLEGAL_INSTRUCTION','CUDA_ERROR_LAUNCH_FAILED','illegal instruction','unspecified launch failure')), message
    print('EXPECTED_RLD_TRAP')
else:
    raise AssertionError('invalid run length decode input did not trap')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EXPECTED_RLD_TRAP" in result.stdout


@pytest.mark.parametrize("offset_dtype", [np.uint32, np.uint64])
def test_untyped_auxiliary_outputs_adopt_selected_dtype(offset_dtype):
    @cuda.jit
    def kernel(output, relative_output, total_output):
        block = numba_coop.this_block()
        values = numba_coop.ThreadData(1, dtype=np.int32)
        lengths = numba_coop.ThreadData(1, dtype=np.uint32)
        values[0] = 7
        lengths[0] = 3 if cuda.threadIdx.x == 0 else 0
        total = numba_coop.ThreadData(1)
        relative = numba_coop.ThreadData(2)
        decoded = numba_coop.run_length_decode(
            block,
            values,
            lengths,
            decoded_items_per_thread=2,
            decoded_offset_dtype=offset_dtype,
            total_decoded_size=total,
            relative_offsets=relative,
        )
        # Scalar indexing must work without a later Store inferring aux dtypes.
        for item in range(2):
            index = cuda.threadIdx.x * 2 + item
            output[index] = decoded[item]
            relative_output[index] = relative[item]
        total_output[cuda.threadIdx.x] = total[0]

    output = np.empty(64, dtype=np.int32)
    relative = np.empty(64, dtype=offset_dtype)
    total = np.empty(32, dtype=offset_dtype)
    kernel[1, 32](output, relative, total)
    cuda.synchronize()
    np.testing.assert_array_equal(output[:3], [7, 7, 7])
    np.testing.assert_array_equal(relative[:3], [0, 1, 2])
    assert not output[3:].any()
    assert np.all(relative[3:] == np.iinfo(offset_dtype).max)
    assert np.all(total == 3)
