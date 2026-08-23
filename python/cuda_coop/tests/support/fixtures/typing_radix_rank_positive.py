# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict positive typing fixture for group-first Radix Rank."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import assert_type

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    class CompilerInteger:
        """Structural compiler integer accepted as a qualified scalar key."""

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
    compiler_keys = CompilerPayload()

    assert_type(coop.radix_rank(block, int_keys), coop.ThreadDataLike[int])
    assert_type(
        coop.radix_rank(block, int32_keys, begin_bit=np.int32(28)),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.radix_rank(
            block,
            uint32_keys,
            begin_bit=np.uint32(24),
            radix_bits=np.uint32(8),
            descending=True,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.radix_rank(block, int64_keys, begin_bit=60, end_bit=64),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.radix_rank(block, uint64_keys),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.radix_rank(block, compiler_keys),
        coop.ThreadDataLike[int],
    )

    cutlass_block = cutlass_coop.this_block()
    cutlass_keys = cutlass_coop.ThreadData.from_values(
        np.int64(-1),
        np.int64(1),
    )
    cutlass_prefix = cutlass_coop.ThreadData.from_values(
        np.int32(0),
        np.int32(0),
    )
    assert_type(
        cutlass_coop.radix_rank(
            cutlass_block,
            cutlass_keys,
            begin_bit=np.uint32(60),
            end_bit=64,
            descending=True,
            exclusive_digit_prefix=cutlass_prefix,
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(cutlass_coop.radix_rank(cutlass_block, 3), int)
    assert_type(
        cutlass_coop.radix_rank(cutlass_block, np.uint64(3)),
        int,
    )
    assert_type(
        cutlass_coop.radix_rank(cutlass_block, CompilerInteger()),
        int,
    )
    assert_type(
        cutlass_coop.radix_rank(cutlass_block, RegisterTensor()),
        cutlass_coop.ThreadData[int],
    )

    numba_block = numba_coop.this_block()
    numba_keys = numba_coop.ThreadData(2, np.int32)
    numba_prefix = numba_coop.ThreadData(2, np.int32)
    assert_type(
        numba_coop.radix_rank(
            numba_block,
            numba_keys,
            begin_bit=np.int32(28),
            radix_bits=4,
            exclusive_digit_prefix=numba_prefix,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(numba_coop.radix_rank(numba_block, 3), int)
    assert_type(numba_coop.radix_rank(numba_block, np.int64(-3)), int)
    assert_type(numba_coop.radix_rank(numba_block, CompilerInteger()), int)

    # Qualified groups and payloads also satisfy the common portable overload.
    assert_type(
        coop.radix_rank(cutlass_block, cutlass_keys),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.radix_rank(numba_block, numba_keys),
        coop.ThreadDataLike[int],
    )
