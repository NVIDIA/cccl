# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict CUTLASS-installed typing fixture for register-payload collectives."""

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING, Any

from typing_extensions import assert_type

if TYPE_CHECKING:
    from cutlass.cute import Tensor, TensorSSA

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop

    common_block = coop.this_block()
    qualified_block = cutlass_coop.this_block()
    qualified_warp = cutlass_coop.this_warp()

    class KnownRegisterTensor:
        """Structurally qualified register tensor without ThreadData methods."""

        @property
        def element_type(self) -> object:
            return object()

        @property
        def shape(self) -> object:
            return (2,)

        @property
        def memspace(self) -> object:
            return object()

        def load(self) -> object:
            return object()

    def accepts_qualified_register_payload(payload: Tensor | TensorSSA) -> None:
        """Exercise every qualified transform that adapts a CUTLASS tensor."""

        assert_type(
            cutlass_coop.scan(qualified_block, payload),
            cutlass_coop.ThreadData[Any],
        )
        assert_type(
            cutlass_coop.exclusive_sum(qualified_block, payload),
            cutlass_coop.ThreadData[Any],
        )
        assert_type(
            cutlass_coop.inclusive_sum(qualified_block, payload),
            cutlass_coop.ThreadData[Any],
        )
        assert_type(
            cutlass_coop.exclusive_scan(qualified_block, payload),
            cutlass_coop.ThreadData[Any],
        )
        assert_type(
            cutlass_coop.inclusive_scan(qualified_block, payload),
            cutlass_coop.ThreadData[Any],
        )
        assert_type(
            cutlass_coop.exchange(qualified_block, payload),
            cutlass_coop.ThreadData[Any],
        )
        assert_type(
            cutlass_coop.exchange(qualified_warp, payload),
            cutlass_coop.ThreadData[Any],
        )
        assert_type(
            cutlass_coop.adjacent_difference(qualified_block, payload),
            cutlass_coop.ThreadData[Any],
        )
        assert_type(
            cutlass_coop.discontinuity(qualified_block, payload),
            cutlass_coop.ThreadData[int],
        )
        assert_type(
            cutlass_coop.discontinuity(
                qualified_block,
                payload,
                mode="heads_and_tails",
            ),
            tuple[cutlass_coop.ThreadData[int], cutlass_coop.ThreadData[int]],
        )
        assert_type(
            cutlass_coop.shuffle(qualified_block, payload),
            cutlass_coop.ThreadData[Any],
        )
        assert_type(
            cutlass_coop.shuffle(qualified_block, payload, mode="up"),
            cutlass_coop.ThreadData[Any],
        )
        assert_type(
            cutlass_coop.store(qualified_block, object(), payload),
            None,
        )

    known_register = KnownRegisterTensor()
    assert_type(
        cutlass_coop.exchange(qualified_block, known_register),
        cutlass_coop.ThreadData[Any],
    )
    coop.reduce(common_block, known_register)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.sum(common_block, known_register)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.scan(common_block, known_register)  # pyright: ignore[reportCallIssue, reportArgumentType]
    coop.exclusive_sum(  # pyright: ignore[reportCallIssue]
        common_block,
        known_register,  # pyright: ignore[reportArgumentType]
    )
    coop.inclusive_sum(  # pyright: ignore[reportCallIssue]
        common_block,
        known_register,  # pyright: ignore[reportArgumentType]
    )
    coop.exclusive_scan(  # pyright: ignore[reportCallIssue]
        common_block,
        known_register,  # pyright: ignore[reportArgumentType]
    )
    coop.inclusive_scan(  # pyright: ignore[reportCallIssue]
        common_block,
        known_register,  # pyright: ignore[reportArgumentType]
    )
    coop.exchange(
        common_block,
        known_register,  # pyright: ignore[reportArgumentType]
    )
    coop.adjacent_difference(
        common_block,
        known_register,  # pyright: ignore[reportArgumentType]
    )
    coop.discontinuity(
        common_block,
        known_register,  # pyright: ignore[reportArgumentType]
    )
    coop.shuffle(
        common_block,
        known_register,  # pyright: ignore[reportArgumentType]
    )
    coop.store(  # pyright: ignore[reportCallIssue]
        common_block,
        object(),
        known_register,  # pyright: ignore[reportArgumentType]
    )
