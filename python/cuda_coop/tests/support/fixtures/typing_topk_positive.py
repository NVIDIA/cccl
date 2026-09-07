# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict positive typing fixture for group-first TopK."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import assert_type

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    class CompilerInteger:
        """Structural compiler integer usable as a key or uniform control."""

        width: int = 32
        signed: bool = True

        @property
        def dtype(self) -> object:
            return object()

        def ir_value(self) -> object:
            return object()

    class CompilerIntegerPayload:
        """Static stand-in for compiler-owned integral ThreadData."""

        items_per_thread: int = 2
        dtype: object | None = None

        def __len__(self) -> int:
            return self.items_per_thread

        def __getitem__(self, index: int, /) -> CompilerInteger:
            return CompilerInteger()

        def __setitem__(
            self,
            index: int,
            value: CompilerInteger,
            /,
        ) -> None:
            pass

    class RegisterTensor:
        """Structural CUTLASS register-memory tensor stand-in."""

        @property
        def element_type(self) -> object:
            return object()

        @property
        def shape(self) -> object:
            return object()

        @property
        def memspace(self) -> object:
            return object()

        def load(self) -> object:
            return object()

    block = coop.this_block()
    int_keys = coop.ThreadData(2, int)
    int32_keys = coop.ThreadData(2, np.int32)
    uint32_keys = coop.ThreadData(2, np.uint32)
    int64_keys = coop.ThreadData(2, np.int64)
    uint64_keys = coop.ThreadData(2, np.uint64)
    float_values = coop.ThreadData(2, np.float64)
    compiler_keys = CompilerIntegerPayload()
    compiler_control = CompilerInteger()
    portable_storage = coop.TempStorage(size_in_bytes=16_400, alignment=16)

    assert_type(
        coop.topk_max_keys(
            block,
            int_keys,
            7,
            temp_storage=portable_storage,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.topk_min_keys(
            block,
            int32_keys,
            np.int32(7),
            valid_items=np.uint32(63),
            begin_bit=compiler_control,
            end_bit=np.int32(32),
        ),
        coop.ThreadDataLike[np.int32],
    )
    assert_type(
        coop.topk_max_keys(block, uint32_keys, compiler_control),
        coop.ThreadDataLike[np.uint32],
    )
    assert_type(
        coop.topk_min_keys(block, int64_keys, np.int64(7)),
        coop.ThreadDataLike[np.int64],
    )
    assert_type(
        coop.topk_max_keys(block, uint64_keys, np.uint64(7)),
        coop.ThreadDataLike[np.uint64],
    )
    assert_type(
        coop.topk_min_keys(block, compiler_keys, compiler_control),
        coop.ThreadDataLike[CompilerInteger],
    )
    max_pair_keys, max_pair_values = coop.topk_max_pairs(
        block, uint32_keys, float_values, compiler_control
    )
    min_pair_keys, min_pair_values = coop.topk_min_pairs(
        block, int64_keys, float_values, np.int32(1)
    )
    assert_type(max_pair_keys, coop.ThreadDataLike[np.uint32])
    assert_type(max_pair_values, coop.ThreadDataLike[np.float64])
    assert_type(min_pair_keys, coop.ThreadDataLike[np.int64])
    assert_type(min_pair_values, coop.ThreadDataLike[np.float64])

    cutlass_block = cutlass_coop.this_block()
    cutlass_keys = cutlass_coop.ThreadData.from_values(
        np.uint8(3),
        np.uint8(1),
    )
    cutlass_values = cutlass_coop.ThreadData.from_values(
        np.float64(30),
        np.float64(10),
    )
    cutlass_storage = cutlass_coop.TempStorage(
        size_in_bytes=16_400,
        alignment=16,
    )
    assert_type(
        cutlass_coop.topk_max_keys(
            cutlass_block,
            cutlass_keys,
            np.int32(7),
            valid_items=63,
            begin_bit=0,
            end_bit=np.int32(8),
            temp_storage=cutlass_storage,
        ),
        cutlass_coop.ThreadData[np.uint8],
    )
    assert_type(
        cutlass_coop.topk_min_keys(cutlass_block, np.float32(3), 7),
        np.float32,
    )
    assert_type(
        cutlass_coop.topk_max_pairs(
            cutlass_block,
            cutlass_keys,
            cutlass_values,
            compiler_control,
        ),
        tuple[
            cutlass_coop.ThreadData[np.uint8],
            cutlass_coop.ThreadData[np.float64],
        ],
    )
    assert_type(
        cutlass_coop.topk_min_pairs(
            cutlass_block,
            np.int32(3),
            np.float32(30),
            7,
        ),
        tuple[np.int32, np.float32],
    )
    assert_type(
        cutlass_coop.topk_max_keys(cutlass_block, RegisterTensor(), 7),
        cutlass_coop.ThreadData[Any],
    )
    assert_type(
        cutlass_coop.topk_min_pairs(
            cutlass_block,
            RegisterTensor(),
            RegisterTensor(),
            7,
        ),
        tuple[cutlass_coop.ThreadData[Any], cutlass_coop.ThreadData[Any]],
    )

    numba_block = numba_coop.this_block()
    numba_int8_keys = numba_coop.ThreadData(2, np.int8)
    numba_float16_keys = numba_coop.ThreadData(2, np.float16)
    numba_bool_values = numba_coop.ThreadData(2, np.bool_)
    numba_storage = numba_coop.TempStorage(size_in_bytes=16_400, alignment=16)
    assert_type(
        numba_coop.topk_max_keys(
            numba_block,
            numba_int8_keys,
            np.int32(7),
            valid_items=63,
            temp_storage=numba_storage,
        ),
        coop.ThreadDataLike[np.int8],
    )
    assert_type(
        numba_coop.topk_min_keys(numba_block, numba_float16_keys, 7),
        coop.ThreadDataLike[np.float16],
    )
    assert_type(
        numba_coop.topk_max_pairs(
            numba_block,
            numba_float16_keys,
            numba_bool_values,
            compiler_control,
            begin_bit=np.int32(0),
            end_bit=np.int32(16),
        ),
        tuple[
            coop.ThreadDataLike[np.float16],
            coop.ThreadDataLike[np.bool_],
        ],
    )

    external_dtype: object = object()
    external_keys = numba_coop.ThreadData(2, external_dtype)
    assert_type(
        numba_coop.topk_min_keys(numba_block, external_keys, 7),
        coop.ThreadDataLike[Any],
    )

    # Qualified groups and conservative payloads satisfy the common contract.
    assert_type(
        coop.topk_max_keys(cutlass_block, cutlass_coop.ThreadData.from_values(3, 1), 7),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.topk_min_keys(
            numba_block,
            numba_coop.ThreadData(2, np.uint64),
            7,
        ),
        coop.ThreadDataLike[np.uint64],
    )
