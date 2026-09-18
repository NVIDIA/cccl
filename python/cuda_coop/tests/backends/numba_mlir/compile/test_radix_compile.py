# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""GPU-free production compiler coverage for radix operations."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")
from numba_cuda_mlir import cuda, tools, types

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]


@pytest.fixture(autouse=True)
def _hidden_device(monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    monkeypatch.setattr(
        tools,
        "get_gpu_compute_capability",
        lambda as_type=str: (9, 0) if as_type is tuple else "sm_90",
    )
    monkeypatch.setattr(
        cuda, "get_current_device", lambda: SimpleNamespace(compute_capability=(9, 0))
    )


def _compile(kernel, signature):
    result = kernel._compile_launch_config_signature(
        signature,
        (
            ("grid", (1, 1, 1)),
            ("block", (64, 1, 1)),
            ("sharedmem", 0),
            ("cluster", None),
        ),
    )
    assert result.metadata["cubin"]
    return result.metadata["mlir_module_str"]


@pytest.mark.parametrize("operation", ["keys", "pairs", "rank"])
@pytest.mark.parametrize(
    "dtype", [types.int32, types.uint32, types.int64, types.uint64]
)
def test_radix_compiles_and_preserves_result_dtype(operation, dtype):
    @cuda.jit(chip="sm_90")
    def kernel(source, destination, associated, begin, end):
        t = cuda.threadIdx.x
        keys = coop.ThreadData(2, dtype=dtype)
        keys[0] = source[t * 2]
        keys[1] = source[t * 2 + 1]
        if operation == "keys":
            result = coop.radix_sort_keys(
                coop.this_block(), keys, begin_bit=begin, end_bit=end
            )
            destination[t * 2] = result[0]
        elif operation == "pairs":
            values = coop.ThreadData(2, dtype=types.float64)
            values[0] = associated[t * 2]
            values[1] = associated[t * 2 + 1]
            result, payload = coop.radix_sort_pairs(
                coop.this_block(), keys, values, temp_storage=coop.TempStorage()
            )
            destination[t * 2] = result[0]
            associated[t * 2] = payload[0]
        else:
            ranks = coop.radix_rank(coop.this_block(), keys)
            # Composition must use int32 rank dtype, even with uint64 keys.
            ranked = coop.radix_sort_keys(coop.this_block(), ranks)
            destination[t * 2] = ranked[0]

    mlir = _compile(
        kernel,
        types.void(
            dtype[::1], dtype[::1], types.float64[::1], types.int32, types.int32
        ),
    )
    assert "gpu.barrier" in mlir


def test_qualified_prefix_and_float_striped_sort_compile():
    @cuda.jit(chip="sm_90")
    def kernel(source, ranks_out, prefixes, floating, output):
        t = cuda.threadIdx.x
        keys = numba_coop.ThreadData(2, dtype=types.int32)
        keys[0] = source[t * 2]
        keys[1] = source[t * 2 + 1]
        prefix = numba_coop.ThreadData(2, dtype=types.int32)
        ranks = numba_coop.radix_rank(
            numba_coop.this_block(), keys, radix_bits=7, exclusive_digit_prefix=prefix
        )
        ranks_out[t] = ranks[0]
        prefixes[t] = prefix[0]
        local = cuda.local.array(2, dtype=types.float32)
        local[0] = floating[t * 2]
        local[1] = floating[t * 2 + 1]
        result = numba_coop.radix_sort_keys(
            numba_coop.this_block(), local, blocked_to_striped=True, descending=True
        )
        output[t] = result[0]

    _compile(
        kernel,
        types.void(
            types.int32[::1],
            types.int32[::1],
            types.int32[::1],
            types.float32[::1],
            types.float32[::1],
        ),
    )


@pytest.mark.parametrize(
    "failure", ["warp", "float", "width", "bool", "extent", "prefix"]
)
def test_invalid_radix_contracts_fail_before_device_code(failure):
    @cuda.jit(chip="sm_90")
    def kernel(source, destination):
        keys = coop.ThreadData(
            2, dtype=types.float32 if failure == "float" else types.int32
        )
        keys[0] = source[0]
        keys[1] = source[1]
        if failure == "warp":
            result = coop.radix_sort_keys(coop.this_warp(), keys)
        elif failure == "width":
            result = coop.radix_rank(coop.this_block(), keys, radix_bits=9)
        elif failure == "bool":
            result = coop.radix_sort_keys(coop.this_block(), keys, descending=1)
        elif failure == "extent":
            values = coop.ThreadData(3, dtype=types.int32)
            result, values = coop.radix_sort_pairs(coop.this_block(), keys, values)
        elif failure == "prefix":
            prefix = numba_coop.ThreadData(2, dtype=types.int32)
            result = numba_coop.radix_rank(
                numba_coop.this_block(), keys, exclusive_digit_prefix=prefix
            )
        else:
            result = coop.radix_sort_keys(coop.this_block(), keys)
        destination[0] = result[0]

    with pytest.raises(
        (TypeError, ValueError, NotImplementedError),
        match="block|dtype|int32|width|bool|items_per_thread|items per thread",
    ):
        _compile(kernel, types.void(types.int32[::1], types.int32[::1]))


def test_unsigned_64_bit_runtime_controls_are_rejected():
    @cuda.jit(chip="sm_90")
    def kernel(source, destination, begin, end):
        keys = coop.ThreadData(2, dtype=types.int32)
        keys[0] = source[0]
        keys[1] = source[1]
        result = coop.radix_sort_keys(
            coop.this_block(), keys, begin_bit=begin, end_bit=end
        )
        destination[0] = result[0]

    with pytest.raises(TypeError, match="unsigned integer up to 32 bits"):
        _compile(
            kernel,
            types.void(types.int32[::1], types.int32[::1], types.uint64, types.uint64),
        )
