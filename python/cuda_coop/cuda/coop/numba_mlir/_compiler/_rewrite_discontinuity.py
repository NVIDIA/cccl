# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Discontinuity factory and runtime-argument finalization.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    ir,
)


class _DiscontinuityRewrite:
    def _finalize_discontinuity_factory_kwargs(
        self,
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from .._lowering._discontinuity import BlockDiscontinuityType

        discontinuity_type = factory_kwargs.get(
            "block_discontinuity_type",
            BlockDiscontinuityType.HEADS,
        )
        try:
            discontinuity_type = BlockDiscontinuityType(discontinuity_type)
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'discontinuity' block_discontinuity_type "
                "must be a BlockDiscontinuityType value."
            ) from exc

        has_predecessor = "tile_predecessor_item" in seen_factory_kwargs
        has_successor = "tile_successor_item" in seen_factory_kwargs
        if discontinuity_type is BlockDiscontinuityType.HEADS:
            if has_successor:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'discontinuity' HEADS does not accept "
                    "tile_successor_item."
                )
            if runtime_arg_count == 3 and not has_predecessor:
                factory_kwargs["tile_predecessor_item"] = True
                seen_factory_kwargs.add("tile_predecessor_item")
                has_predecessor = True
            expected_count = 2 + int(has_predecessor)
        elif discontinuity_type is BlockDiscontinuityType.TAILS:
            if has_predecessor:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'discontinuity' TAILS does not accept "
                    "tile_predecessor_item."
                )
            if runtime_arg_count == 3 and not has_successor:
                factory_kwargs["tile_successor_item"] = True
                seen_factory_kwargs.add("tile_successor_item")
                has_successor = True
            expected_count = 2 + int(has_successor)
        else:
            if runtime_arg_count == 5:
                if not has_predecessor:
                    factory_kwargs["tile_predecessor_item"] = True
                    seen_factory_kwargs.add("tile_predecessor_item")
                    has_predecessor = True
                if not has_successor:
                    factory_kwargs["tile_successor_item"] = True
                    seen_factory_kwargs.add("tile_successor_item")
                    has_successor = True
            elif runtime_arg_count == 4:
                if has_predecessor and has_successor:
                    raise CoopSinglePhaseRewriteError(
                        "coop single-phase 'discontinuity' cannot map four "
                        "runtime arguments to both boundary items."
                    )
                if not has_predecessor and not has_successor:
                    factory_kwargs["tile_predecessor_item"] = True
                    seen_factory_kwargs.add("tile_predecessor_item")
                    has_predecessor = True
            expected_count = 3 + int(has_predecessor) + int(has_successor)
        if runtime_arg_count != expected_count:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'discontinuity' runtime argument count "
                f"{runtime_arg_count} does not match {expected_count}."
            )

    @staticmethod
    def _reorder_discontinuity_runtime_args(
        runtime_args: list[ir.Var],
        factory_kwargs: dict[str, object],
    ) -> list[ir.Var]:
        from .._lowering._discontinuity import BlockDiscontinuityType

        discontinuity_type = BlockDiscontinuityType(
            factory_kwargs.get(
                "block_discontinuity_type",
                BlockDiscontinuityType.HEADS,
            )
        )
        if discontinuity_type in {
            BlockDiscontinuityType.HEADS,
            BlockDiscontinuityType.TAILS,
        }:
            if len(runtime_args) < 2:
                return runtime_args
            return [runtime_args[1], runtime_args[0], *runtime_args[2:]]
        if len(runtime_args) < 3:
            return runtime_args

        input_items, head_flags, tail_flags = runtime_args[:3]
        has_predecessor = "tile_predecessor_item" in factory_kwargs
        has_successor = "tile_successor_item" in factory_kwargs
        if has_predecessor and has_successor:
            return [
                head_flags,
                runtime_args[3],
                tail_flags,
                runtime_args[4],
                input_items,
            ]
        if has_predecessor:
            return [head_flags, runtime_args[3], tail_flags, input_items]
        if has_successor:
            return [head_flags, tail_flags, runtime_args[3], input_items]
        return [head_flags, tail_flags, input_items]


__all__ = ["_DiscontinuityRewrite"]
