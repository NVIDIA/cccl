# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Adjacent-difference factory finalization.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
)


class _AdjacentDifferenceRewrite:
    def _finalize_adjacent_difference_factory_kwargs(
        self,
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from .._lowering._adjacent_difference import (
            BlockAdjacentDifferenceType,
        )

        adjacent_type = factory_kwargs.get(
            "block_adjacent_difference_type",
            BlockAdjacentDifferenceType.SubtractLeft,
        )
        try:
            adjacent_type = BlockAdjacentDifferenceType(adjacent_type)
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' "
                "block_adjacent_difference_type must be a "
                "BlockAdjacentDifferenceType value."
            ) from exc

        if adjacent_type is BlockAdjacentDifferenceType.SubtractLeft:
            tile_kw = "tile_predecessor_item"
            invalid_tile_kw = "tile_successor_item"
        else:
            tile_kw = "tile_successor_item"
            invalid_tile_kw = "tile_predecessor_item"
        if invalid_tile_kw in seen_factory_kwargs:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' received invalid "
                f"'{invalid_tile_kw}' for {adjacent_type.name}."
            )

        has_valid_items = "valid_items" in seen_factory_kwargs
        has_boundary = tile_kw in seen_factory_kwargs
        if runtime_arg_count == 4:
            if not has_valid_items:
                factory_kwargs["valid_items"] = True
                seen_factory_kwargs.add("valid_items")
                has_valid_items = True
            if not has_boundary:
                factory_kwargs[tile_kw] = True
                seen_factory_kwargs.add(tile_kw)
                has_boundary = True
        elif runtime_arg_count == 3:
            if has_valid_items and has_boundary:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'adjacent_difference' cannot map three "
                    "runtime arguments to both valid_items and a boundary item."
                )
            if not has_valid_items and not has_boundary:
                factory_kwargs["valid_items"] = True
                seen_factory_kwargs.add("valid_items")
                has_valid_items = True
        elif runtime_arg_count != 2:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' expects two, three, "
                "or four runtime arguments."
            )

        if (
            adjacent_type is BlockAdjacentDifferenceType.SubtractRight
            and has_valid_items
            and has_boundary
        ):
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' cannot combine a "
                "right partial tile with tile_successor_item."
            )
        expected_count = 2 + int(has_valid_items) + int(has_boundary)
        if runtime_arg_count != expected_count:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'adjacent_difference' runtime argument "
                f"count {runtime_arg_count} does not match {expected_count}."
            )


__all__ = ["_AdjacentDifferenceRewrite"]
