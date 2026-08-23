# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict positive typing fixture for group-first Radix Sort."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import assert_type

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    class CompilerInteger:
        """Structural compiler integer usable as a key or bit control."""

        width: int = 32
        signed: bool = True

        @property
        def dtype(self) -> object:
            return object()

        def ir_value(self) -> object:
            return object()

    class CompilerPayload:
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
        """Structural CUTLASS register-memory Tensor stand-in."""

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
    float_values = coop.ThreadData(2, np.float32)
    compiler_keys = CompilerPayload()
    compiler_bit = CompilerInteger()

    assert_type(
        coop.radix_sort_keys(block, int_keys),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.radix_sort_keys(block, int32_keys, begin_bit=4),
        coop.ThreadDataLike[np.int32],
    )
    assert_type(
        coop.radix_sort_keys(block, uint32_keys, begin_bit=4, end_bit=None),
        coop.ThreadDataLike[np.uint32],
    )
    assert_type(
        coop.radix_sort_keys(
            block,
            int64_keys,
            begin_bit=np.uint32(8),
            end_bit=compiler_bit,
            descending=True,
            temp_storage=coop.TempStorage(),
        ),
        coop.ThreadDataLike[np.int64],
    )
    assert_type(
        coop.radix_sort_keys(block, uint64_keys),
        coop.ThreadDataLike[np.uint64],
    )
    assert_type(
        coop.radix_sort_keys(block, compiler_keys, begin_bit=compiler_bit),
        coop.ThreadDataLike[CompilerInteger],
    )
    pair_keys, pair_values = coop.radix_sort_pairs(
        block,
        uint64_keys,
        float_values,
        begin_bit=compiler_bit,
    )
    assert_type(pair_keys, coop.ThreadDataLike[np.uint64])
    assert_type(pair_values, coop.ThreadDataLike[np.float32])

    cutlass_block = cutlass_coop.this_block()
    cutlass_keys = cutlass_coop.ThreadData.from_values(np.uint32(3), np.uint32(1))
    cutlass_values = cutlass_coop.ThreadData.from_values(
        np.float64(30),
        np.float64(10),
    )
    assert_type(
        cutlass_coop.radix_sort_pairs(
            cutlass_block,
            cutlass_keys,
            cutlass_values,
            begin_bit=np.int32(4),
            end_bit=CompilerInteger(),
        ),
        tuple[
            cutlass_coop.ThreadData[np.uint32],
            cutlass_coop.ThreadData[np.float64],
        ],
    )
    assert_type(
        cutlass_coop.radix_sort_pairs(
            cutlass_block,
            np.uint32(3),
            np.float64(30),
        ),
        tuple[np.uint32, np.float64],
    )
    assert_type(
        cutlass_coop.radix_sort_pairs(
            cutlass_block,
            RegisterTensor(),
            RegisterTensor(),
        ),
        tuple[cutlass_coop.ThreadData[Any], cutlass_coop.ThreadData[Any]],
    )
    assert_type(
        cutlass_coop.radix_sort_keys(
            cutlass_block,
            cutlass_keys,
            begin_bit=4,
            end_bit=None,
            descending=True,
            temp_storage=cutlass_coop.TempStorage(),
        ),
        cutlass_coop.ThreadData[np.uint32],
    )
    assert_type(cutlass_coop.radix_sort_keys(cutlass_block, 3), int)
    assert_type(
        cutlass_coop.radix_sort_keys(cutlass_block, np.int64(3)),
        np.int64,
    )
    assert_type(
        cutlass_coop.radix_sort_keys(cutlass_block, RegisterTensor()),
        cutlass_coop.ThreadData[Any],
    )

    numba_block = numba_coop.this_block()
    numba_keys = numba_coop.ThreadData(2, np.int64)
    numba_values = numba_coop.ThreadData(2, np.float32)
    assert_type(
        numba_coop.radix_sort_pairs(
            numba_block,
            numba_keys,
            numba_values,
            begin_bit=np.int32(3),
            end_bit=CompilerInteger(),
        ),
        tuple[coop.ThreadDataLike[np.int64], coop.ThreadDataLike[np.float32]],
    )
    assert_type(
        numba_coop.radix_sort_pairs(
            numba_block,
            np.uint64(3),
            np.float64(30),
        ),
        tuple[np.uint64, np.float64],
    )
    assert_type(
        numba_coop.radix_sort_keys(
            numba_block,
            numba_keys,
            begin_bit=np.int32(3),
            end_bit=None,
            temp_storage=numba_coop.TempStorage(),
        ),
        coop.ThreadDataLike[np.int64],
    )
    assert_type(numba_coop.radix_sort_keys(numba_block, np.uint64(3)), np.uint64)

    # Qualified groups and concrete CUTLASS payloads still satisfy the common
    # portable contract; only the common overload set remains visible here.
    assert_type(
        coop.radix_sort_keys(cutlass_block, cutlass_keys),
        coop.ThreadDataLike[np.uint32],
    )
    assert_type(
        coop.radix_sort_keys(numba_block, numba_keys),
        coop.ThreadDataLike[np.int64],
    )
