# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shuffle factory finalization.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
)


class _ShuffleRewrite:
    def _finalize_shuffle_factory_kwargs(
        self,
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from cuda.coop.numba_mlir._lowering._shuffle import (
            BlockShuffleType,
            _normalize_shuffle_type,
        )

        try:
            shuffle_type = _normalize_shuffle_type(
                factory_kwargs.get("block_shuffle_type", BlockShuffleType.Up)
            )
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(str(exc)) from exc
        if runtime_arg_count == 1:
            if "items_per_thread" in seen_factory_kwargs:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase scalar 'shuffle' does not accept items_per_thread."
                )
            if (
                "block_prefix" in seen_factory_kwargs
                or "block_suffix" in seen_factory_kwargs
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase scalar 'shuffle' does not accept block_prefix/block_suffix."
                )
            if "distance" not in seen_factory_kwargs:
                factory_kwargs["distance"] = 1
                seen_factory_kwargs.add("distance")
            return
        if runtime_arg_count not in {2, 3}:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'shuffle' runtime argument count must be one of {1, 2, 3}."
            )
        if shuffle_type not in {BlockShuffleType.Up, BlockShuffleType.Down}:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase array 'shuffle' only supports BlockShuffleType.Up/Down."
            )
        if "distance" in seen_factory_kwargs:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase array 'shuffle' does not support distance."
            )
        if runtime_arg_count == 2:
            if (
                "block_prefix" in seen_factory_kwargs
                or "block_suffix" in seen_factory_kwargs
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase array 'shuffle' received block_prefix/block_suffix without a matching runtime boundary argument."
                )
            return
        if shuffle_type == BlockShuffleType.Up:
            if "block_prefix" in seen_factory_kwargs:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase array 'shuffle' with BlockShuffleType.Up does not support block_prefix."
                )
            if "block_suffix" not in seen_factory_kwargs:
                factory_kwargs["block_suffix"] = True
                seen_factory_kwargs.add("block_suffix")
        else:
            if "block_suffix" in seen_factory_kwargs:
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase array 'shuffle' with BlockShuffleType.Down does not support block_suffix."
                )
            if "block_prefix" not in seen_factory_kwargs:
                factory_kwargs["block_prefix"] = True
                seen_factory_kwargs.add("block_prefix")


__all__ = ["_ShuffleRewrite"]
