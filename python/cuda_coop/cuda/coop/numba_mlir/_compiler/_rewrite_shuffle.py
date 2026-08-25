# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shuffle factory finalization.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_payload import PayloadInference
from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    ir,
)


class _ShuffleRewrite:
    def _infer_shuffle_payload(self, inference: PayloadInference) -> None:
        """Infer scalar or array shuffle payload metadata."""

        if len(inference.runtime_args) == 1:
            value = inference.runtime_args[0]
            inferred_dtype = (
                self._resolve_var_dtype(value) if isinstance(value, ir.Var) else None
            )
            inference.infer_kwarg(
                "dtype", inferred_dtype or inference.factory_value("dtype")
            )
            return
        input_var, input_spec = inference.candidate(0)
        output_var, output_spec = inference.candidate(1)
        self._require_matching_items_per_thread(
            inference.op_name, "input", input_spec, "output", output_spec
        )
        extent = input_spec.items_per_thread if input_spec is not None else None
        if extent is None and output_spec is not None:
            extent = output_spec.items_per_thread
        inference.infer_kwarg("items_per_thread", extent)
        inferred_dtype = input_spec.dtype if input_spec is not None else None
        if inferred_dtype is None and output_spec is not None:
            inferred_dtype = output_spec.dtype
        if inferred_dtype is None and input_var is not None:
            inferred_dtype = self._resolve_var_dtype(input_var)
        inference.infer_kwarg(
            "dtype", inferred_dtype or inference.factory_value("dtype")
        )
        if inferred_dtype is not None:
            if input_var is not None:
                self._record_inferred_thread_data_dtype(input_var, inferred_dtype)
            if output_var is not None:
                self._record_inferred_thread_data_dtype(output_var, inferred_dtype)

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
