# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict CUTLASS-installed typing fixture for Merge Sort compiler values."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING, Any

from typing_extensions import assert_type

if TYPE_CHECKING:
    from cutlass.base_dsl.typing import Float32, Int32
    from cutlass.cute import Tensor, TensorSSA

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop

    integer = Int32(3)
    keys = cutlass_coop.ThreadData.from_values(integer, Int32(1))
    common_block = coop.this_block()
    qualified_block = cutlass_coop.this_block()
    qualified_warp = cutlass_coop.this_warp()

    assert_type(
        coop.merge_sort_keys(
            common_block,
            keys,
            valid_items=integer,
            oob_default=Int32(0),
        ),
        coop.ThreadDataLike[Int32],
    )
    assert_type(
        cutlass_coop.merge_sort_keys(qualified_block, integer),
        Int32,
    )
    assert_type(
        cutlass_coop.merge_sort_keys(
            qualified_warp,
            keys,
            valid_items=integer,
            oob_default=Int32(0),
        ),
        cutlass_coop.ThreadData[Int32],
    )

    def accepts_register_tensors(tensor: Tensor, tensor_ssa: TensorSSA) -> None:
        assert_type(
            cutlass_coop.merge_sort_keys(qualified_block, tensor),
            cutlass_coop.ThreadData[Any],
        )
        assert_type(
            cutlass_coop.merge_sort_keys(
                qualified_warp,
                tensor_ssa,
                valid_items=integer,
                oob_default=Int32(0),
            ),
            cutlass_coop.ThreadData[Any],
        )
        assert_type(
            cutlass_coop.merge_sort_pairs(qualified_block, tensor, tensor_ssa),
            tuple[cutlass_coop.ThreadData[Any], cutlass_coop.ThreadData[Any]],
        )
        assert_type(
            cutlass_coop.radix_sort_pairs(qualified_block, tensor, tensor_ssa),
            tuple[cutlass_coop.ThreadData[Any], cutlass_coop.ThreadData[Any]],
        )
        assert_type(
            cutlass_coop.merge_sort_pairs(
                qualified_warp,
                tensor_ssa,
                tensor,
                valid_items=integer,
                oob_default=Int32(0),
            ),
            tuple[cutlass_coop.ThreadData[Any], cutlass_coop.ThreadData[Any]],
        )

    cutlass_coop.merge_sort_keys(  # pyright: ignore[reportCallIssue]
        qualified_block,
        keys,
        valid_items=Float32(1),  # pyright: ignore[reportArgumentType]
        oob_default=Int32(0),
    )
    cutlass_coop.merge_sort_keys(qualified_block, Float32(1))  # pyright: ignore[reportCallIssue, reportArgumentType]
