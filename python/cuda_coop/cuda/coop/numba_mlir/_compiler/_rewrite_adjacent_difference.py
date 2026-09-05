# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Adjacent-difference payload inference and factory finalization.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_payload import PayloadInference
from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _dtype_values_match,
)


class _AdjacentDifferenceRewrite:
    def _infer_adjacent_difference_payload(self, inference: PayloadInference) -> None:
        """Infer adjacent-difference payload shape and dtype metadata."""

        input_var, input_spec = inference.candidate(0)
        output_var, output_spec = inference.candidate(1)
        self._require_matching_items_per_thread(
            inference.op_name,
            "input",
            input_spec,
            "output",
            output_spec,
        )
        extent = input_spec.items_per_thread if input_spec is not None else None
        if extent is None and output_spec is not None:
            extent = output_spec.items_per_thread
        inference.infer_kwarg("items_per_thread", extent)

        input_dtype = input_spec.dtype if input_spec is not None else None
        output_dtype = output_spec.dtype if output_spec is not None else None
        if input_dtype is None and input_var is not None:
            input_dtype = self._resolve_var_dtype(input_var)
        if output_dtype is None and output_var is not None:
            output_dtype = self._resolve_var_dtype(output_var)
        if (
            input_dtype is not None
            and output_dtype is not None
            and not _dtype_values_match(input_dtype, output_dtype)
        ):
            raise CoopSinglePhaseRewriteError(
                "coop adjacent_difference input and output dtypes must match."
            )
        inferred_dtype = input_dtype
        if inferred_dtype is None:
            inferred_dtype = output_dtype
        if inferred_dtype is None:
            inferred_dtype = inference.factory_value("dtype")
        inference.infer_kwarg("dtype", inferred_dtype)
        for payload_var in (input_var, output_var):
            if inferred_dtype is not None and payload_var is not None:
                self._record_inferred_thread_data_dtype(
                    payload_var,
                    inferred_dtype,
                )

        boundary_index = 2 + int(bool(inference.factory_kwargs.get("valid_items")))
        boundary_name = None
        if inference.factory_kwargs.get("tile_predecessor_item"):
            boundary_name = "tile_predecessor_item"
        elif inference.factory_kwargs.get("tile_successor_item"):
            boundary_name = "tile_successor_item"
        if boundary_name is not None and boundary_index < len(inference.runtime_args):
            boundary_dtype = self._resolve_var_dtype(
                inference.runtime_args[boundary_index]
            )
            if (
                inferred_dtype is not None
                and boundary_dtype is not None
                and not _dtype_values_match(inferred_dtype, boundary_dtype)
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop adjacent_difference boundary dtype must match "
                    "the input dtype."
                )

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
