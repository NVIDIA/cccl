# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict positive typing fixture for group-first Merge Sort."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

import numpy as np
from typing_extensions import assert_type

if TYPE_CHECKING:
    import cuda.coop.cutlass as cutlass_coop
    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    class CompilerInteger:
        """Structural backend integer used without importing a compiler."""

        width: int = 32
        signed: bool = True

        @property
        def dtype(self) -> object:
            return object()

        def ir_value(self) -> object:
            return object()

    class CompilerPayload:
        """Static stand-in for compiler-owned integer ThreadData."""

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

    common_block = coop.this_block()
    common_warp = coop.this_warp()
    common_logical_warp = common_warp.group_by(8)
    common_int_keys = coop.ThreadData(2, int)
    common_numpy_keys = coop.ThreadData(2, np.uint32)
    common_float_values = coop.ThreadData(2, np.float64)
    compiler_keys = CompilerPayload()
    compiler_count = CompilerInteger()

    assert_type(
        coop.merge_sort_keys(common_block, common_int_keys),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.merge_sort_keys(
            common_warp,
            common_numpy_keys,
            descending=True,
            valid_items=np.int32(48),
            oob_default=np.uint32(0),
            temp_storage=coop.TempStorage(),
        ),
        coop.ThreadDataLike[np.uint32],
    )
    assert_type(
        coop.merge_sort_keys(
            common_block,
            compiler_keys,
            valid_items=compiler_count,
            oob_default=CompilerInteger(),
        ),
        coop.ThreadDataLike[CompilerInteger],
    )
    common_pair_keys, common_pair_values = coop.merge_sort_pairs(
        common_warp,
        common_numpy_keys,
        common_float_values,
    )
    assert_type(common_pair_keys, coop.ThreadDataLike[np.uint32])
    assert_type(common_pair_values, coop.ThreadDataLike[np.float64])
    assert_type(
        coop.merge_sort_keys(common_logical_warp, common_int_keys),
        coop.ThreadDataLike[int],
    )

    cutlass_block = cutlass_coop.this_block()
    cutlass_warp = cutlass_coop.this_warp()
    cutlass_logical_warp = cutlass_warp.group_by(16)
    cutlass_keys = cutlass_coop.ThreadData.from_values(3, 1)
    assert_type(
        cutlass_coop.merge_sort_keys(cutlass_block, cutlass_keys),
        cutlass_coop.ThreadData[int],
    )
    assert_type(
        cutlass_coop.merge_sort_keys(
            cutlass_warp,
            cutlass_keys,
            valid_items=np.uint32(31),
            oob_default=0,
            compare_op="greater",
        ),
        cutlass_coop.ThreadData[int],
    )
    cutlass_float_keys = cutlass_coop.ThreadData.from_values(
        np.float32(3),
        np.float32(1),
    )
    assert_type(
        cutlass_coop.merge_sort_keys(cutlass_warp, cutlass_float_keys),
        cutlass_coop.ThreadData[np.float32],
    )
    cutlass_uint8_keys = cutlass_coop.ThreadData.from_values(
        np.uint8(3),
        np.uint8(1),
    )
    cutlass_values = cutlass_coop.ThreadData.from_values(
        np.float64(30),
        np.float64(10),
    )
    assert_type(
        cutlass_coop.merge_sort_pairs(
            cutlass_block,
            cutlass_keys,
            cutlass_values,
        ),
        tuple[
            cutlass_coop.ThreadData[int],
            cutlass_coop.ThreadData[np.float64],
        ],
    )
    assert_type(
        cutlass_coop.merge_sort_pairs(
            cutlass_warp,
            cutlass_float_keys,
            cutlass_values,
            valid_items=np.int32(31),
            oob_default=np.float32(0),
        ),
        tuple[
            cutlass_coop.ThreadData[np.float32],
            cutlass_coop.ThreadData[np.float64],
        ],
    )
    assert_type(
        cutlass_coop.merge_sort_pairs(cutlass_block, 3, np.float64(30)),
        tuple[int, np.float64],
    )
    assert_type(
        cutlass_coop.merge_sort_keys(
            cutlass_logical_warp,
            cutlass_uint8_keys,
            valid_items=24,
            oob_default=np.uint8(0),
        ),
        cutlass_coop.ThreadData[np.uint8],
    )
    assert_type(
        cutlass_coop.merge_sort_keys(
            cutlass_logical_warp,
            cutlass_keys,
        ),
        cutlass_coop.ThreadData[int],
    )
    assert_type(cutlass_coop.merge_sort_keys(cutlass_block, 3), int)
    assert_type(
        cutlass_coop.merge_sort_keys(
            cutlass_block,
            np.int32(3),
            valid_items=64,
            oob_default=np.int32(0),
        ),
        np.int32,
    )

    def integer_less(left: int, right: int) -> bool:
        return left < right

    def float_less(left: np.float32, right: np.float32) -> bool:
        return bool(left < right)

    numba_block = numba_coop.this_block()
    numba_warp = numba_coop.this_warp()
    numba_logical_warp = numba_warp.group_by(8)
    numba_keys = numba_coop.ThreadData(2, int)
    numba_float_keys = numba_coop.ThreadData(2, np.float32)
    numba_float16_keys = numba_coop.ThreadData(2, np.float16)
    numba_values = numba_coop.ThreadData(2, np.float64)
    assert_type(
        numba_coop.merge_sort_pairs(numba_block, numba_keys, numba_values),
        tuple[coop.ThreadDataLike[int], coop.ThreadDataLike[np.float64]],
    )
    assert_type(
        numba_coop.merge_sort_pairs(
            numba_warp,
            np.float32(3),
            np.int16(30),
            valid_items=np.int32(31),
            oob_default=np.float32(0),
        ),
        tuple[np.float32, np.int16],
    )
    assert_type(
        numba_coop.merge_sort_keys(
            numba_block,
            numba_keys,
            compare_op=integer_less,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(numba_coop.merge_sort_keys(numba_block, 3), int)
    assert_type(
        numba_coop.merge_sort_keys(
            numba_warp,
            np.float32(3),
            valid_items=29,
            oob_default=np.float32(0),
            compare_op=float_less,
        ),
        np.float32,
    )
    assert_type(
        numba_coop.merge_sort_keys(
            numba_warp,
            numba_float_keys,
            valid_items=np.uint32(29),
            oob_default=np.float32(0),
            compare_op=float_less,
        ),
        coop.ThreadDataLike[np.float32],
    )
    assert_type(
        numba_coop.merge_sort_keys(numba_block, numba_float16_keys),
        coop.ThreadDataLike[np.float16],
    )
    assert_type(
        numba_coop.merge_sort_keys(
            numba_warp,
            numba_keys,
            valid_items=np.uint32(29),
            oob_default=0,
        ),
        coop.ThreadDataLike[int],
    )
    assert_type(
        numba_coop.merge_sort_keys(numba_logical_warp, numba_keys),
        coop.ThreadDataLike[int],
    )

    # Qualified groups and CUTLASS ThreadData remain valid portable inputs.
    assert_type(
        coop.merge_sort_keys(numba_warp, common_int_keys),
        coop.ThreadDataLike[int],
    )
    assert_type(
        coop.merge_sort_keys(cutlass_block, cutlass_keys),
        coop.ThreadDataLike[int],
    )
