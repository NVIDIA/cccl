# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Discontinuity payload inference and factory/runtime finalization.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_payload import PayloadInference
from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _dtype_values_match,
    ir,
)


class _DiscontinuityRewrite:
    def _infer_discontinuity_payload(self, inference: PayloadInference) -> None:
        """Infer discontinuity input and flag payload metadata."""

        from .._lowering._discontinuity import BlockDiscontinuityType

        mode = BlockDiscontinuityType(
            inference.factory_kwargs.get(
                "block_discontinuity_type",
                BlockDiscontinuityType.HEADS,
            )
        )
        input_var, input_spec = inference.candidate(0)
        head_var, head_spec = inference.candidate(1)
        tail_var, tail_spec = (
            inference.candidate(2)
            if mode is BlockDiscontinuityType.HEADS_AND_TAILS
            else (None, None)
        )
        self._require_matching_items_per_thread(
            inference.op_name,
            "input",
            input_spec,
            "head flags",
            head_spec,
        )
        self._require_matching_items_per_thread(
            inference.op_name,
            "input",
            input_spec,
            "tail flags",
            tail_spec,
        )
        extent = input_spec.items_per_thread if input_spec is not None else None
        if extent is None and head_spec is not None:
            extent = head_spec.items_per_thread
        if extent is None and tail_spec is not None:
            extent = tail_spec.items_per_thread
        inference.infer_kwarg("items_per_thread", extent)

        inferred_dtype = input_spec.dtype if input_spec is not None else None
        if inferred_dtype is None and input_var is not None:
            inferred_dtype = self._resolve_var_dtype(input_var)
        if inferred_dtype is None:
            inferred_dtype = inference.factory_value("dtype")
        inference.infer_kwarg("dtype", inferred_dtype)

        from numba_cuda_mlir import types as numba_mlir_types

        flag_dtype = numba_mlir_types.int32
        inference.infer_kwarg("flag_dtype", flag_dtype)
        for flag_name, flag_var, flag_spec in (
            ("head", head_var, head_spec),
            ("tail", tail_var, tail_spec),
        ):
            if flag_var is None:
                continue
            actual_flag_dtype = flag_spec.dtype if flag_spec is not None else None
            if actual_flag_dtype is None:
                actual_flag_dtype = self._resolve_var_dtype(flag_var)
            if actual_flag_dtype is not None and not _dtype_values_match(
                actual_flag_dtype,
                flag_dtype,
            ):
                raise CoopSinglePhaseRewriteError(
                    f"coop discontinuity {flag_name} flags must use int32 dtype."
                )
            self._record_inferred_thread_data_dtype(flag_var, flag_dtype)
        if inferred_dtype is not None and input_var is not None:
            self._record_inferred_thread_data_dtype(input_var, inferred_dtype)

        boundary_index = 3 if mode is BlockDiscontinuityType.HEADS_AND_TAILS else 2
        for boundary_name in (
            "tile_predecessor_item",
            "tile_successor_item",
        ):
            if not inference.factory_kwargs.get(boundary_name):
                continue
            if boundary_index >= len(inference.runtime_args):
                break
            boundary_dtype = self._resolve_var_dtype(
                inference.runtime_args[boundary_index]
            )
            if (
                inferred_dtype is not None
                and boundary_dtype is not None
                and not _dtype_values_match(inferred_dtype, boundary_dtype)
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop discontinuity boundary dtype must match the input dtype."
                )
            boundary_index += 1

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
