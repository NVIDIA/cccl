# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

import cuda.coop.numba_mlir as coop

_THREADS = 64
_GLOBAL_SCALAR = np.int32(3)
_MODULE_BLOCK = coop.this_block()


def _kernel_for(binary_op, algorithm="warp_reductions"):
    @cuda.jit
    def kernel(source, output):
        thread = cuda.threadIdx.x
        result = coop.reduce(
            coop.this_block(),
            source[thread],
            binary_op=binary_op,
            algorithm=algorithm,
        )
        if thread == 0:
            output[0] = result

    return kernel


@cuda.jit
def _default_reduce(source, output):
    thread = cuda.threadIdx.x
    result = coop.reduce(coop.this_block(), source[thread])
    if thread == 0:
        output[0] = result


def test_default_reduce_is_sum():
    source = np.arange(_THREADS, dtype=np.int32)
    output = np.zeros(1, dtype=np.int32)

    _default_reduce[1, _THREADS](source, output)

    assert output[0] == source.sum(dtype=np.int32)


@cuda.jit
def _sum_with_module_block(source, output):
    thread = cuda.threadIdx.x
    result = coop.sum(_MODULE_BLOCK, source[thread])
    if thread == 0:
        output[0] = result


def test_module_level_block_descriptor_is_a_compile_time_constant():
    source = np.arange(32, dtype=np.int32)
    output = np.zeros(1, dtype=np.int32)

    _sum_with_module_block[1, 32](source, output)

    assert output[0] == source.sum(dtype=np.int32)


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
def test_sum_supports_portable_scalar_numeric_types(dtype):
    source = np.ones(32, dtype=dtype)
    output = np.zeros(1, dtype=dtype)

    _default_reduce[1, 32](source, output)

    assert output[0] == 32


@cuda.jit
def _literal_sum(output):
    result = coop.sum(coop.this_block(), 1)
    if cuda.threadIdx.x == 0:
        output[0] = result


def test_literal_scalar_is_normalized_to_its_numeric_type():
    output = np.zeros(1, dtype=np.int64)

    _literal_sum[1, 32](output)

    assert output[0] == 32


@pytest.mark.parametrize(
    ("binary_op", "source", "expected"),
    [
        ("sum", np.arange(_THREADS, dtype=np.int32), sum(range(_THREADS))),
        (
            "multiplies",
            np.array([2, 3, *([1] * (_THREADS - 2))], dtype=np.int32),
            6,
        ),
        ("min", np.arange(_THREADS, dtype=np.int32) - 17, -17),
        ("max", np.arange(_THREADS, dtype=np.int32) - 17, _THREADS - 18),
        ("bit_and", np.full(_THREADS, 0x7F, dtype=np.int32), 0x7F),
        ("bit_or", np.arange(_THREADS, dtype=np.int32), 0x3F),
        (
            "bit_xor",
            np.arange(_THREADS, dtype=np.int32),
            np.bitwise_xor.reduce(np.arange(_THREADS, dtype=np.int32)),
        ),
    ],
)
def test_builtin_reduction_operators(binary_op, source, expected):
    output = np.zeros(1, dtype=source.dtype)

    _kernel_for(binary_op)[1, _THREADS](source, output)

    assert output[0] == expected


@pytest.mark.parametrize(
    "algorithm",
    ["raking_commutative_only", "raking", "warp_reductions"],
)
def test_block_reduce_algorithms(algorithm):
    source = np.linspace(-1.0, 1.0, _THREADS, dtype=np.float32)
    output = np.zeros(1, dtype=np.float32)

    _kernel_for("max", algorithm)[1, _THREADS](source, output)

    assert output[0] == pytest.approx(1.0)


@cuda.jit
def _runtime_valid_prefix(source, output, valid_items):
    thread = cuda.threadIdx.x
    result = coop.sum(
        coop.this_block(),
        source[thread],
        valid_items=valid_items,
    )
    if thread == 0:
        output[0] = result


def test_runtime_valid_prefix_is_root_only():
    source = np.arange(_THREADS, dtype=np.uint64)
    output = np.zeros(1, dtype=np.uint64)
    valid_items = np.int64(37)

    _runtime_valid_prefix[1, _THREADS](source, output, valid_items)

    assert output[0] == source[:valid_items].sum(dtype=np.uint64)


@cuda.jit
def _static_valid_prefix(source, output):
    thread = cuda.threadIdx.x
    result = coop.sum(
        coop.this_block(),
        source[thread],
        valid_items=17,
    )
    if thread == 0:
        output[0] = result


def test_static_valid_prefix_is_lowered_through_the_portable_plan():
    source = np.arange(32, dtype=np.int32)
    output = np.zeros(1, dtype=np.int32)

    _static_valid_prefix[1, 32](source, output)

    assert output[0] == source[:17].sum(dtype=np.int32)


@cuda.jit
def _back_to_back_reductions(source, output):
    thread = cuda.threadIdx.x
    first = coop.sum(coop.this_block(), source[thread])
    second = coop.reduce(
        coop.this_block(),
        source[thread] + 1,
        binary_op="max",
    )
    if thread == 0:
        output[0] = first
        output[1] = second


def test_back_to_back_reductions_reuse_provider_storage_safely():
    source = np.arange(32, dtype=np.int32)
    output = np.zeros(2, dtype=np.int32)

    _back_to_back_reductions[1, 32](source, output)

    np.testing.assert_array_equal(
        output,
        np.array([source.sum(dtype=np.int32), source.max() + 1], dtype=np.int32),
    )


@cuda.jit
def _looped_reduction(source, output):
    thread = cuda.threadIdx.x
    value = source[thread]
    for iteration in range(3):
        result = coop.sum(coop.this_block(), value + iteration)
        if thread == 0:
            output[iteration] = result


def test_looped_reduction_reuses_provider_storage_safely():
    source = np.arange(32, dtype=np.int32)
    output = np.zeros(3, dtype=np.int32)

    _looped_reduction[1, 32](source, output)

    base = source.sum(dtype=np.int32)
    np.testing.assert_array_equal(
        output,
        np.array([base, base + 32, base + 64], dtype=np.int32),
    )


@cuda.jit
def _two_dimensional_block(source, output):
    thread = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x
    result = coop.sum(coop.this_block(), source[thread])
    if thread == 0:
        output[0] = result


def test_exact_multidimensional_launch_shapes_the_cub_specialization():
    source = np.arange(32, dtype=np.int32)
    output = np.zeros(1, dtype=np.int32)

    _two_dimensional_block[1, (8, 4)](source, output)

    assert output[0] == source.sum(dtype=np.int32)


@cuda.jit
def _overload_typed_payloads(source, output):
    thread = cuda.threadIdx.x
    global_thread = cuda.grid(1)
    thread_result = coop.sum(coop.this_block(), thread)
    grid_result = coop.sum(coop.this_block(), global_thread)
    global_result = coop.sum(coop.this_block(), _GLOBAL_SCALAR)
    promoted_result = coop.sum(coop.this_block(), source[thread] + 1)
    if thread == 0:
        output[0] = thread_result
        output[1] = grid_result
        output[2] = global_result
        output[3] = promoted_result


def test_overload_typing_selects_each_payload_provider():
    source = np.arange(32, dtype=np.int32)
    output = np.zeros(4, dtype=np.int64)

    _overload_typed_payloads[1, 32](source, output)

    np.testing.assert_array_equal(
        output,
        np.array(
            [
                sum(range(32)),
                sum(range(32)),
                32 * int(_GLOBAL_SCALAR),
                source.sum(dtype=np.int64) + 32,
            ],
            dtype=np.int64,
        ),
    )
