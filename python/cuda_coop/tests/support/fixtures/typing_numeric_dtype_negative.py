# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict negative fixture for the portable numeric dtype closure."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    class CustomValue:
        pass

    class ComplexProducer:
        def __cuda_coop_thread_data_load__(self) -> complex:
            return 1 + 2j

    common_block = coop.this_block()
    python_complex: complex = 1 + 2j
    custom_value = CustomValue()
    opaque_value: object = object()
    numpy_int8 = np.int8(1)
    numpy_uint16 = np.uint16(1)
    numpy_float16 = np.float16(1)
    numpy_complex = np.complex64(1 + 2j)

    coop.sum(common_block, python_complex)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.scan(common_block, python_complex)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.sum(common_block, custom_value)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.scan(common_block, opaque_value)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.sum(common_block, numpy_int8)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.scan(common_block, numpy_uint16)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.sum(common_block, numpy_float16)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.scan(common_block, numpy_complex)  # pyright: ignore[reportCallIssue, reportArgumentType]

    coop.ThreadData(2, complex)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.ThreadData(2, object)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.ThreadData(2, object())  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.ThreadData(2, CustomValue)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.ThreadData(2, np.int8)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.ThreadData(2, np.uint16)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.ThreadData(2, np.float16)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.ThreadData(2, np.complex64)  # pyright: ignore[reportCallIssue, reportArgumentType]

    common_payload = coop.ThreadData(2, int)
    common_block.rank_as(complex)  # pyright: ignore[reportCallIssue, reportArgumentType]
    common_block.count_as(CustomValue)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.load(  # pyright: ignore[reportCallIssue]
        common_block,
        object(),
        common_payload,
        valid_items=1,
        oob_default=python_complex,  # pyright: ignore[reportArgumentType]
    )
    coop.store(common_block, object(), python_complex)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.store(  # pyright: ignore[reportCallIssue]
        common_block,
        object(),
        common_payload,
        valid_items=python_complex,  # pyright: ignore[reportArgumentType]
        offset=python_complex,  # pyright: ignore[reportArgumentType]
    )
    coop.scan(common_block, 1, initial_value=python_complex)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.exclusive_scan(common_block, 1, initial_value=python_complex)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.adjacent_difference(
        common_block,
        common_payload,
        tile_predecessor_item=python_complex,  # pyright: ignore[reportArgumentType]
    )
    coop.discontinuity(
        common_block,
        common_payload,
        tile_successor_item=python_complex,  # pyright: ignore[reportArgumentType]
    )

    cutlass_block = cutlass_coop.this_block()
    cutlass_coop.sum(cutlass_block, python_complex)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.scan(cutlass_block, python_complex)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.sum(cutlass_block, custom_value)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.scan(cutlass_block, opaque_value)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.sum(cutlass_block, numpy_int8)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.scan(cutlass_block, numpy_float16)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.scan(cutlass_block, numpy_complex)  # pyright: ignore[reportCallIssue, reportArgumentType]

    cutlass_coop.ThreadData(2, object())  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_payload = cutlass_coop.ThreadData.from_values(1, 2)
    cutlass_complex_payload = cutlass_coop.ThreadData.from_values(python_complex)
    cutlass_coop.sum(cutlass_block, cutlass_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.scan(cutlass_block, cutlass_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_payload.to_tensor_ssa(dtype=complex)  # pyright: ignore[reportArgumentType]
    cutlass_payload.to_register_tensor(dtype=CustomValue)  # pyright: ignore[reportArgumentType]
    cutlass_block.rank_as(complex)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_block.count_as(CustomValue)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.load(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        object(),
        cutlass_payload,
        valid_items=1,
        oob_default=python_complex,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.store(cutlass_block, object(), python_complex)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.store(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        object(),
        cutlass_payload,
        valid_items=python_complex,  # pyright: ignore[reportArgumentType]
        offset=python_complex,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.scan(cutlass_block, 1, initial_value=python_complex)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.exclusive_scan(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        1,
        initial_value=python_complex,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.adjacent_difference(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_payload,
        tile_predecessor_item=python_complex,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.discontinuity(  # pyright: ignore[reportCallIssue]
        cutlass_block,
        cutlass_payload,
        tile_successor_item=python_complex,  # pyright: ignore[reportArgumentType]
    )
    cutlass_coop.ThreadData.from_register_tensor(  # pyright: ignore[reportCallIssue]
        object(),  # pyright: ignore[reportArgumentType]
        dtype=complex,
    )
    cutlass_coop.ThreadData.from_vector(  # pyright: ignore[reportCallIssue]
        object(),  # pyright: ignore[reportArgumentType]
        dtype=np.float16,
    )
    cutlass_coop.ThreadData.from_payload(  # pyright: ignore[reportCallIssue]
        object(),  # pyright: ignore[reportArgumentType]
        dtype=CustomValue,
    )
    cutlass_coop.ThreadData.from_register_tensor(python_complex)  # pyright: ignore[reportArgumentType]
    cutlass_coop.ThreadData.from_register_tensor(custom_value)  # pyright: ignore[reportArgumentType]
    cutlass_coop.ThreadData.from_vector(python_complex)  # pyright: ignore[reportArgumentType]
    cutlass_coop.ThreadData.from_vector(custom_value)  # pyright: ignore[reportArgumentType]
    cutlass_coop.ThreadData.from_payload(python_complex)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.ThreadData.from_payload(custom_value)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.ThreadData.load(
        ComplexProducer(),  # pyright: ignore[reportArgumentType]
    )

    class IndexOnlyVector:
        def __getitem__(self, index: int, /) -> np.int32:
            return np.int32(index)

    class IndexOnlyRegisterTensor:
        @property
        def memspace(self) -> object:
            return object()

        def __getitem__(self, index: int, /) -> np.int32:
            return np.int32(index)

    class IndexOnlyProducer:
        def __cuda_coop_thread_data_load__(self) -> IndexOnlyVector:
            return IndexOnlyVector()

    cutlass_coop.ThreadData.from_vector(IndexOnlyVector())  # pyright: ignore[reportArgumentType]
    cutlass_coop.ThreadData.from_payload(IndexOnlyVector())  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.ThreadData.from_register_tensor(IndexOnlyRegisterTensor())  # pyright: ignore[reportArgumentType]
    cutlass_coop.ThreadData.load(IndexOnlyProducer())  # pyright: ignore[reportArgumentType]

    # A backend-qualified Numba payload remains valid in that backend, but its
    # complex item type is outside the portable root and CUTLASS closures.
    numba_complex_payload = numba_coop.ThreadData(2, complex)
    coop.load(common_block, object(), numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.store(common_block, object(), numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.reduce(common_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.sum(common_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.scan(common_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.exclusive_sum(common_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.inclusive_sum(common_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.exclusive_scan(common_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.inclusive_scan(common_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.exchange(common_block, numba_complex_payload)  # pyright: ignore[reportArgumentType]
    coop.adjacent_difference(common_block, numba_complex_payload)  # pyright: ignore[reportArgumentType]
    coop.discontinuity(common_block, numba_complex_payload)  # pyright: ignore[reportArgumentType]
    coop.shuffle(common_block, numba_complex_payload)  # pyright: ignore[reportArgumentType]

    cutlass_coop.load(cutlass_block, object(), numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.store(cutlass_block, object(), numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.reduce(cutlass_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.sum(cutlass_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.scan(cutlass_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.exclusive_sum(cutlass_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.inclusive_sum(cutlass_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.exclusive_scan(cutlass_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.inclusive_scan(cutlass_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.exchange(cutlass_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.adjacent_difference(cutlass_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.discontinuity(cutlass_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
    cutlass_coop.shuffle(cutlass_block, numba_complex_payload)  # pyright: ignore[reportCallIssue, reportArgumentType]
