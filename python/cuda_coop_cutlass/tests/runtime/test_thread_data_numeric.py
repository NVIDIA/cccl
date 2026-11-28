# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check numeric casts before register export and dtype selector adaptation."""

import cutlass
import numpy as np
import pytest
from cutlass import cute

import cuda.coop.cutlass as coop
from tests.runtime_support import device_array

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize(
    "source_type,destination_type,source_value,numpy_type",
    [
        (cutlass.Uint8, cutlass.Int32, 200, np.int32),
        (cutlass.Uint16, cutlass.Uint32, 60000, np.uint32),
        (cutlass.Uint32, cutlass.Uint64, 0xFFFFFFFF, np.uint64),
        (cutlass.Uint32, cutlass.Float64, 3000000000, np.float64),
        (cutlass.Float32, cutlass.Uint32, 3000000000, np.uint32),
    ],
)
def test_export_signedness(
    source_type, destination_type, source_value, numpy_type
):
    @cute.kernel
    def kernel(destination: cute.Pointer, value: source_type):
        payload = coop.ThreadData.from_values(value, dtype=source_type)
        immutable = payload.to_tensor_ssa(dtype=destination_type)
        mutable = payload.to_register_tensor(dtype=destination_type)
        output = cute.make_tensor(destination, cute.make_layout(2))
        output[0] = immutable[0]
        output[1] = mutable[0]

    @cute.jit
    def launch(destination: cute.Pointer, value: source_type):
        kernel(destination, value).launch(grid=1, block=1)

    output = np.zeros(2, dtype=numpy_type)
    with device_array(output) as destination:
        launch(destination, source_type(source_value))
    np.testing.assert_array_equal(output, np.full(2, source_value, numpy_type))


@pytest.mark.parametrize(
    "selector", [cutlass.Int64, np.int64, np.dtype("int64"), int]
)
def test_casting_constructors_accept_runtime_values(selector):
    source_value = 7 if selector is int else 0xFFFFFFFF

    @cute.kernel
    def kernel(destination: cute.Pointer, value: cutlass.Uint32):
        payload = coop.ThreadData.from_fn(1, lambda _: value, dtype=selector)
        registers = cute.make_rmem_tensor(1, cutlass.Uint32)
        registers[0] = value
        from_registers = coop.ThreadData.from_register_tensor(
            registers, dtype=selector
        )
        from_vector = coop.ThreadData.from_vector(
            registers.load(), dtype=selector
        )
        from_payload = coop.ThreadData.from_payload(
            coop.ThreadData.from_values(value), dtype=selector
        )
        output = cute.make_tensor(destination, cute.make_layout(4))
        output[0] = payload.to_tensor_ssa(dtype=cutlass.Int64)[0]
        output[1] = from_registers.to_tensor_ssa(dtype=cutlass.Int64)[0]
        output[2] = from_vector.to_tensor_ssa(dtype=cutlass.Int64)[0]
        output[3] = from_payload.to_tensor_ssa(dtype=cutlass.Int64)[0]

    @cute.jit
    def launch(destination: cute.Pointer, value: cutlass.Uint32):
        kernel(destination, value).launch(grid=1, block=1)

    output = np.zeros(4, dtype=np.int64)
    with device_array(output) as destination:
        launch(destination, cutlass.Uint32(source_value))
    np.testing.assert_array_equal(output, [source_value] * 4)


@pytest.mark.parametrize("numpy_values", [False, True])
def test_numpy_scalar_values_and_nested_shape_export(numpy_values):
    shape = (1, 2) if numpy_values else (np.int64(1), np.int32(2))
    first = np.int32(1) if numpy_values else 1
    second = np.int32(2) if numpy_values else 2

    @cute.kernel
    def kernel(destination: cute.Pointer):
        payload = coop.ThreadData.from_values(first, second, dtype=np.int32)
        vector = payload.to_tensor_ssa(shape=shape)
        registers = coop.ThreadData.from_fn(
            2, lambda i: i + 3, dtype=np.dtype("int32")
        ).to_register_tensor(shape=shape)
        output = cute.make_tensor(destination, cute.make_layout(4))
        output[0] = vector[0]
        output[1] = vector[1]
        output[2] = registers[0]
        output[3] = registers[1]

    @cute.jit
    def launch(destination: cute.Pointer):
        kernel(destination).launch(grid=1, block=1)

    output = np.zeros(4, dtype=np.int32)
    with device_array(output) as destination:
        launch(destination)
    np.testing.assert_array_equal(output, [1, 2, 3, 4])


class _ReadonlyOnes:
    dtype = cutlass.Int32
    items_per_thread = 2

    def __len__(self):
        return self.items_per_thread

    def __getitem__(self, index):
        return cutlass.Int32((1, 1)[index])


@pytest.mark.parametrize(
    "operation", ("store", "exchange", "shuffle", "scan", "sum")
)
def test_readonly_collective_input(operation):
    @cute.kernel
    def kernel(destination: cute.Pointer):
        block = coop.this_block()
        values = _ReadonlyOnes()
        if cutlass.const_expr(operation == "store"):
            coop.store(block, destination, values)
        elif cutlass.const_expr(operation == "exchange"):
            coop.store(block, destination, coop.exchange(block, values))
        elif cutlass.const_expr(operation == "shuffle"):
            result = coop.shuffle(block, values, mode="down")
            if cute.arch.thread_idx()[0] == 31:
                result[1] = cutlass.Int32(0)
            coop.store(block, destination, result)
        elif cutlass.const_expr(operation == "scan"):
            coop.store(block, destination, coop.inclusive_sum(block, values))
        else:
            total = coop.sum(block, values)
            if cute.arch.thread_idx()[0] == 0:
                cute.make_tensor(destination, cute.make_layout(1))[0] = total

    @cute.jit
    def launch(destination: cute.Pointer):
        kernel(destination).launch(grid=1, block=32)

    observed = np.zeros(64, dtype=np.int32)
    with device_array(observed) as destination:
        launch(destination)
    expected = np.ones_like(observed)
    if operation == "sum":
        expected[:] = 0
        expected[0] = 64
    elif operation == "scan":
        expected = np.arange(1, 65)
    elif operation == "shuffle":
        expected[-1] = 0
    np.testing.assert_array_equal(observed, expected)
