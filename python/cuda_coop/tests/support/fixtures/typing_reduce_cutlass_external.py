# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict CUTLASS-installed typing fixture for Reduce compiler scalars."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING, Any

from typing_extensions import assert_type

if TYPE_CHECKING:
    from cutlass.base_dsl.typing import Float32, Int32
    from cutlass.cute import Tensor, TensorSSA

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop

    value = Int32(1)
    common_block = coop.this_block()
    qualified_block = cutlass_coop.this_block()

    assert_type(
        coop.reduce(
            common_block,
            value,
            broadcast=False,
            valid_items=value,
        ),
        Int32,
    )

    def accepts_register_tensors(tensor: Tensor, tensor_ssa: TensorSSA) -> None:
        assert_type(cutlass_coop.reduce(qualified_block, tensor), Any)
        assert_type(cutlass_coop.reduce(qualified_block, tensor_ssa), Any)
        assert_type(cutlass_coop.sum(qualified_block, tensor), Any)
        assert_type(cutlass_coop.sum(qualified_block, tensor_ssa), Any)

    assert_type(
        cutlass_coop.reduce(
            qualified_block,
            value,
            broadcast=False,
            valid_items=value,
        ),
        Int32,
    )
    assert_type(
        cutlass_coop.sum(
            qualified_block,
            value,
            broadcast=False,
            algorithm="raking",
        ),
        Int32,
    )
    cutlass_coop.reduce(  # pyright: ignore[reportCallIssue]
        qualified_block,
        value,
        broadcast=False,
        valid_items=Float32(1),  # pyright: ignore[reportArgumentType]
    )
