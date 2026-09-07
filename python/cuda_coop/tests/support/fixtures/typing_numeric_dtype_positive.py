# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict positive fixture for portable and qualified numeric dtype closure."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import assert_type

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    python_int: int = 1
    python_float: float = 1.0
    numpy_uint8 = np.uint8(1)
    numpy_int32 = np.int32(1)
    numpy_uint32 = np.uint32(1)
    numpy_int64 = np.int64(1)
    numpy_uint64 = np.uint64(1)
    numpy_float32 = np.float32(1)
    numpy_float64 = np.float64(1)

    common_block = coop.this_block()
    assert_type(coop.sum(common_block, python_int), int)
    assert_type(coop.scan(common_block, python_float), float)
    assert_type(coop.sum(common_block, numpy_uint8), np.uint8)
    assert_type(coop.scan(common_block, numpy_int32), np.int32)
    assert_type(coop.sum(common_block, numpy_uint32), np.uint32)
    assert_type(coop.scan(common_block, numpy_int64), np.int64)
    assert_type(coop.sum(common_block, numpy_uint64), np.uint64)
    assert_type(coop.scan(common_block, numpy_float32), np.float32)
    assert_type(coop.sum(common_block, numpy_float64), np.float64)

    assert_type(coop.ThreadData(2, int), coop.ThreadDataLike[int])
    assert_type(coop.ThreadData(2, float), coop.ThreadDataLike[float])
    assert_type(coop.ThreadData(2, np.uint8), coop.ThreadDataLike[np.uint8])
    assert_type(coop.ThreadData(2, np.int32), coop.ThreadDataLike[np.int32])
    assert_type(coop.ThreadData(2, np.uint32), coop.ThreadDataLike[np.uint32])
    assert_type(coop.ThreadData(2, np.int64), coop.ThreadDataLike[np.int64])
    assert_type(coop.ThreadData(2, np.uint64), coop.ThreadDataLike[np.uint64])
    assert_type(coop.ThreadData(2, np.float32), coop.ThreadDataLike[np.float32])
    assert_type(coop.ThreadData(2, np.float64), coop.ThreadDataLike[np.float64])

    external_dtype: Any = object()
    external_scalar: Any = object()
    assert_type(coop.ThreadData(2, external_dtype), coop.ThreadDataLike[Any])
    assert_type(coop.sum(common_block, external_scalar), Any)
    assert_type(coop.scan(common_block, external_scalar), Any)

    class CompilerScalar:
        width: int = 32

        @property
        def dtype(self) -> object:
            return object()

        def ir_value(self) -> object:
            return object()

    compiler_scalar = CompilerScalar()
    assert_type(common_block.rank_as(CompilerScalar), CompilerScalar)
    assert_type(common_block.count_as(CompilerScalar), CompilerScalar)
    assert_type(common_block.rank_as(external_dtype), Any)
    assert_type(common_block.count_as(external_dtype), Any)
    assert_type(coop.sum(common_block, compiler_scalar), CompilerScalar)
    assert_type(coop.scan(common_block, compiler_scalar), CompilerScalar)

    cutlass_block = cutlass_coop.this_block()
    assert_type(cutlass_block.rank_as(CompilerScalar), CompilerScalar)
    assert_type(cutlass_block.count_as(CompilerScalar), CompilerScalar)
    assert_type(cutlass_block.rank_as(external_dtype), Any)
    assert_type(cutlass_block.count_as(external_dtype), Any)
    assert_type(cutlass_coop.sum(cutlass_block, python_int), int)
    assert_type(cutlass_coop.scan(cutlass_block, python_float), float)
    assert_type(cutlass_coop.sum(cutlass_block, numpy_uint8), np.uint8)
    assert_type(cutlass_coop.scan(cutlass_block, numpy_int32), np.int32)
    assert_type(cutlass_coop.sum(cutlass_block, numpy_uint32), np.uint32)
    assert_type(cutlass_coop.scan(cutlass_block, numpy_int64), np.int64)
    assert_type(cutlass_coop.sum(cutlass_block, numpy_uint64), np.uint64)
    assert_type(cutlass_coop.scan(cutlass_block, numpy_float32), np.float32)
    assert_type(cutlass_coop.sum(cutlass_block, numpy_float64), np.float64)
    assert_type(
        cutlass_coop.ThreadData(2, np.float64),
        cutlass_coop.ThreadData[np.float64],
    )
    assert_type(
        cutlass_coop.ThreadData(2, external_dtype),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.ThreadData(2, external_dtype, values=[1, 2]),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.ThreadData.from_values(numpy_int32, numpy_int32),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(
        cutlass_coop.ThreadData.from_fn(2, lambda index: np.float32(index)),
        cutlass_coop.ThreadData[np.float32],
    )
    python_complex: complex = 1 + 2j
    qualified_complex = cutlass_coop.ThreadData.from_values(python_complex)
    assert_type(qualified_complex, cutlass_coop.ThreadData[complex])
    assert_type(qualified_complex[0], complex)

    class CustomPayload:
        pass

    custom_payload = cutlass_coop.ThreadData.from_values(CustomPayload())
    assert_type(custom_payload, cutlass_coop.ThreadData[CustomPayload])
    assert_type(custom_payload[0], CustomPayload)

    class RegisterTensor:
        @property
        def shape(self) -> object:
            return (2,)

        @property
        def memspace(self) -> object:
            return object()

        def __getitem__(self, index: int, /) -> np.int32:
            return np.int32(index)

    class RegisterVector:
        @property
        def shape(self) -> object:
            return (2,)

        def __getitem__(self, index: int, /) -> np.float32:
            return np.float32(index)

    class NumelVector:
        def numel(self) -> object:
            return 2

        def __getitem__(self, index: int, /) -> np.float32:
            return np.float32(index)

    class IndexOnlyVector:
        def __getitem__(self, index: int, /) -> np.float32:
            return np.float32(index)

    class RegisterProducer:
        def __cuda_coop_thread_data_load__(self) -> RegisterVector:
            return RegisterVector()

    assert_type(
        cutlass_coop.ThreadData.from_register_tensor(RegisterTensor()),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(
        cutlass_coop.ThreadData.from_vector(RegisterVector()),
        cutlass_coop.ThreadData[np.float32],
    )
    assert_type(
        cutlass_coop.ThreadData.from_payload(RegisterVector()),
        cutlass_coop.ThreadData[np.float32],
    )
    assert_type(
        cutlass_coop.ThreadData.from_vector(NumelVector()),
        cutlass_coop.ThreadData[np.float32],
    )
    assert_type(
        cutlass_coop.ThreadData.from_vector(IndexOnlyVector(), items_per_thread=2),
        cutlass_coop.ThreadData[np.float32],
    )
    assert_type(
        cutlass_coop.ThreadData.load(RegisterProducer()),
        cutlass_coop.ThreadData[np.float32],
    )
    # Recognized scalar dtype classes control runtime casts and result types.
    assert_type(
        cutlass_coop.ThreadData.from_fn(
            2,
            lambda index: np.float32(index),
            dtype=int,
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.ThreadData.from_register_tensor(
            RegisterTensor(),
            dtype=float,
        ),
        cutlass_coop.ThreadData[float],
    )
    assert_type(
        cutlass_coop.ThreadData.from_vector(
            RegisterVector(),
            dtype=np.int32,
        ),
        cutlass_coop.ThreadData[np.int32],
    )
    assert_type(
        cutlass_coop.ThreadData.from_payload(
            RegisterVector(),
            dtype=np.float64,
        ),
        cutlass_coop.ThreadData[np.float64],
    )
    assert_type(
        cutlass_coop.ThreadData.load(
            RegisterProducer(),
            dtype=np.int64,
        ),
        cutlass_coop.ThreadData[np.int64],
    )
    # from_values records dtype metadata but does not cast its values.
    assert_type(
        cutlass_coop.ThreadData.from_values(1, dtype=external_dtype),
        cutlass_coop.ThreadData[int],
    )
    # Runtime-casting helpers cannot recover a type from an opaque dtype token.
    assert_type(
        cutlass_coop.ThreadData.from_fn(
            2,
            lambda index: index,
            dtype=external_dtype,
        ),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.ThreadData.from_register_tensor(
            RegisterTensor(),
            dtype=external_dtype,
        ),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.ThreadData.from_vector(
            RegisterVector(),
            dtype=external_dtype,
        ),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.ThreadData.from_payload(
            RegisterVector(),
            dtype=external_dtype,
        ),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.ThreadData.load(
            RegisterProducer(),
            dtype=external_dtype,
        ),
        cutlass_coop.ThreadData[Any],
    )
    external_payload: Any = object()
    assert_type(
        cutlass_coop.ThreadData.from_register_tensor(
            external_payload, items_per_thread=2
        ),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.ThreadData.from_vector(external_payload, items_per_thread=2),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.ThreadData.from_payload(external_payload, items_per_thread=2),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.ThreadData.from_values(compiler_scalar),
        cutlass_coop.ThreadData[CompilerScalar],
    )
    assert_type(
        cutlass_coop.ThreadData(2, CompilerScalar),
        cutlass_coop.ThreadData[CompilerScalar],
    )
    assert_type(
        cutlass_coop.ThreadData.from_values(1).to_register_tensor(
            dtype=external_dtype,
        ),
        object,
    )
    assert_type(cutlass_coop.sum(cutlass_block, compiler_scalar), CompilerScalar)
    assert_type(cutlass_coop.scan(cutlass_block, compiler_scalar), CompilerScalar)

    # Qualified Numba deliberately retains a broader backend-specific numeric
    # surface, including Python and NumPy complex scalar and payload types.
    numba_block = numba_coop.this_block()
    assert_type(numba_block.rank(), np.int32)
    assert_type(numba_block.count(), np.int32)
    assert_type(numba_block.is_member(), np.uint8)
    numpy_complex = np.complex64(1 + 2j)
    assert_type(numba_coop.sum(numba_block, python_complex), complex)
    assert_type(numba_coop.scan(numba_block, python_complex), complex)
    assert_type(numba_coop.sum(numba_block, numpy_complex), np.complex64)
    assert_type(numba_coop.scan(numba_block, numpy_complex), np.complex64)
    assert_type(
        numba_coop.ThreadData(2, complex),
        coop.ThreadDataLike[complex],
    )
    assert_type(
        numba_coop.ThreadData(2, np.complex64),
        coop.ThreadDataLike[np.complex64],
    )
