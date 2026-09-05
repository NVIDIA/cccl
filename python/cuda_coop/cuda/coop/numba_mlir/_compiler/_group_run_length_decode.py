# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run Length Decode IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

from ._group_planner_support import (
    _PAYLOAD_DTYPE_LIKE,
    Any,
    GroupRewriteError,
    Integral,
    ThreadGroup,
    inspect,
    ir,
    operator,
)


class _RunLengthDecodePlanning:
    def _lower_run_length_decode(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        operation = "run_length_decode"
        scope_name = "cuda.coop" if is_common_root else "cuda.coop.numba_mlir"
        self._reject_extra_root_arguments(operation, bound)
        if group.kind != "block":
            raise NotImplementedError(
                f"{scope_name}.run_length_decode currently lowers only "
                "complete physical block groups"
            )

        decoded_items_per_thread = self._constant(
            bound.arguments["decoded_items_per_thread"]
        )
        if isinstance(decoded_items_per_thread, bool) or not isinstance(
            decoded_items_per_thread,
            Integral,
        ):
            raise TypeError(
                f"{scope_name}.run_length_decode decoded_items_per_thread "
                "must be a compile-time positive integer"
            )
        decoded_items_per_thread = int(decoded_items_per_thread)
        if decoded_items_per_thread < 1:
            raise ValueError(
                f"{scope_name}.run_length_decode decoded_items_per_thread "
                "must be a compile-time positive integer"
            )

        offset_is_static, static_offset = self._try_constant(
            bound.arguments["decoded_window_offset"]
        )
        if offset_is_static:
            if isinstance(static_offset, bool) or not isinstance(
                static_offset,
                Integral,
            ):
                raise TypeError(
                    f"{scope_name}.run_length_decode decoded_window_offset "
                    "must be an integer"
                )
            if int(static_offset) < 0:
                raise ValueError(
                    f"{scope_name}.run_length_decode decoded_window_offset "
                    "must be non-negative"
                )

        if is_common_root:
            for name in ("run_values", "run_lengths"):
                if not self._thread_data_operand_state(
                    operation,
                    name,
                    bound.arguments[name],
                ):
                    raise TypeError(
                        f"cuda.coop.run_length_decode requires {name} to be "
                        "coop.ThreadData in the portable API; use "
                        "cuda.coop.numba_mlir for backend-qualified payloads"
                    )

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        run_values = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_values",
            value=bound.arguments["run_values"],
        )
        run_lengths = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_lengths",
            value=bound.arguments["run_lengths"],
        )
        if not self._array_operand_state(
            operation,
            run_values,
        ) or not self._array_operand_state(operation, run_lengths):
            raise TypeError(
                f"{scope_name}.run_length_decode requires fixed-size "
                "ThreadData or local-array run_values and run_lengths payloads"
            )
        runs_per_thread = self._array_extent(run_values)
        if runs_per_thread is None:
            raise GroupRewriteError(
                f"{scope_name}.run_length_decode could not infer runs_per_thread"
            )
        if self._array_extent(run_lengths) != runs_per_thread:
            raise ValueError(
                f"{scope_name}.run_length_decode run_values and run_lengths "
                "must have the same items_per_thread"
            )

        decoded_items = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_decoded",
            prototype=run_values,
            is_array=True,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
            items_per_thread=decoded_items_per_thread,
        )
        total_decoded_size = bound.arguments.get("total_decoded_size")
        if self._is_none(total_decoded_size):
            total_decoded_size = self._typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem="run_length_total_decoded_size",
                prototype=run_lengths,
                is_array=True,
                dtype_policy=_PAYLOAD_DTYPE_LIKE,
                items_per_thread=1,
            )
        else:
            total_decoded_size = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem="run_length_total_decoded_size_output",
                value=total_decoded_size,
            )
            if not self._array_operand_state(operation, total_decoded_size):
                raise TypeError(
                    f"{scope_name}.run_length_decode total_decoded_size "
                    "must be a single-item ThreadData or local-array output"
                )
            if self._array_extent(total_decoded_size) != 1:
                raise ValueError(
                    f"{scope_name}.run_length_decode total_decoded_size "
                    "must contain exactly one item"
                )

        relative_offsets = bound.arguments.get("relative_offsets")
        has_relative_offsets = not self._is_none(relative_offsets)
        if has_relative_offsets:
            relative_offsets = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem="run_length_relative_offsets",
                value=relative_offsets,
            )
            if not self._array_operand_state(operation, relative_offsets):
                raise TypeError(
                    f"{scope_name}.run_length_decode relative_offsets must be "
                    "a ThreadData or local-array output"
                )
            if self._array_extent(relative_offsets) != decoded_items_per_thread:
                raise ValueError(
                    f"{scope_name}.run_length_decode relative_offsets must "
                    "match decoded_items_per_thread"
                )

        decoded_offset_dtype = bound.arguments.get("decoded_offset_dtype")
        assert group.hierarchy is not None
        factory_kwargs: dict[str, Any] = {
            "threads_per_block": group.hierarchy.block_dim,
            "runs_per_thread": runs_per_thread,
            "decoded_items_per_thread": decoded_items_per_thread,
            "with_relative_offsets": has_relative_offsets,
            **({"_common_root_operation": operation} if is_common_root else {}),
        }
        if offset_is_static:
            factory_kwargs["_static_decoded_window_offset"] = int(static_offset)
        if not self._is_none(decoded_offset_dtype):
            factory_kwargs["decoded_offset_dtype"] = decoded_offset_dtype

        mask_index = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_mask_index",
            value=0,
        )
        zero_literal = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_zero_literal",
            value=0,
        )
        statements.append(ir.SetItem(decoded_items, mask_index, zero_literal, loc))
        decoded_zero = self._new_var(scope, loc, "run_length_decoded_zero")
        statements.append(
            ir.Assign(
                ir.Expr.getitem(decoded_items, mask_index, loc),
                decoded_zero,
                loc,
            )
        )
        relative_sentinel = None
        if has_relative_offsets:
            minus_one_literal = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem="run_length_minus_one_literal",
                value=-1,
            )
            statements.append(
                ir.SetItem(relative_offsets, mask_index, minus_one_literal, loc)
            )
            relative_sentinel = self._new_var(
                scope,
                loc,
                "run_length_relative_sentinel",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.getitem(relative_offsets, mask_index, loc),
                    relative_sentinel,
                    loc,
                )
            )

        runtime_args = [
            run_values,
            run_lengths,
            total_decoded_size,
            decoded_items,
        ]
        if has_relative_offsets:
            runtime_args.append(relative_offsets)
        decoded_window_offset = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_decoded_window_offset",
            value=bound.arguments["decoded_window_offset"],
        )
        runtime_args.append(decoded_window_offset)

        from .._lowering._run_length_decode import _group_run_length_decode

        self._emit_factory_call(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_fused",
            factory=_group_run_length_decode,
            args=runtime_args,
            kwargs=factory_kwargs,
        )

        from numba_cuda_mlir import cuda as cuda_module

        rank = self._emit_group_method_call(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_rank",
            group=group,
            operation="rank",
        )
        decoded_items_per_thread_var = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_decoded_items_per_thread",
            value=decoded_items_per_thread,
        )
        rank_base = self._new_var(scope, loc, "run_length_rank_base")
        statements.append(
            ir.Assign(
                ir.Expr.binop(
                    operator.mul,
                    rank,
                    decoded_items_per_thread_var,
                    loc,
                ),
                rank_base,
                loc,
            )
        )
        total_value = self._new_var(scope, loc, "run_length_total")
        statements.append(
            ir.Assign(
                ir.Expr.getitem(total_decoded_size, mask_index, loc),
                total_value,
                loc,
            )
        )
        offset_is_in_range = self._new_var(
            scope,
            loc,
            "run_length_offset_is_in_range",
        )
        statements.append(
            ir.Assign(
                ir.Expr.binop(
                    operator.lt,
                    decoded_window_offset,
                    total_value,
                    loc,
                ),
                offset_is_in_range,
                loc,
            )
        )
        safe_offset = self._emit_factory_call(
            statements,
            scope=scope,
            loc=loc,
            stem="run_length_safe_offset",
            factory=cuda_module.selp,
            args=[offset_is_in_range, decoded_window_offset, total_value],
            kwargs={},
        )
        remaining_items = self._new_var(scope, loc, "run_length_remaining_items")
        statements.append(
            ir.Assign(
                ir.Expr.binop(operator.sub, total_value, safe_offset, loc),
                remaining_items,
                loc,
            )
        )
        for item_index in range(decoded_items_per_thread):
            item_index_var = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"run_length_item_index_{item_index}",
                value=item_index,
            )
            local_target = self._new_var(
                scope,
                loc,
                f"run_length_local_target_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.binop(operator.add, rank_base, item_index_var, loc),
                    local_target,
                    loc,
                )
            )
            is_valid = self._new_var(
                scope,
                loc,
                f"run_length_item_valid_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.binop(operator.lt, local_target, remaining_items, loc),
                    is_valid,
                    loc,
                )
            )
            decoded_value = self._new_var(
                scope,
                loc,
                f"run_length_decoded_value_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.getitem(decoded_items, item_index_var, loc),
                    decoded_value,
                    loc,
                )
            )
            projected = self._emit_factory_call(
                statements,
                scope=scope,
                loc=loc,
                stem=f"run_length_projected_value_{item_index}",
                factory=cuda_module.selp,
                args=[is_valid, decoded_value, decoded_zero],
                kwargs={},
            )
            statements.append(ir.SetItem(decoded_items, item_index_var, projected, loc))
            if has_relative_offsets:
                relative_value = self._new_var(
                    scope,
                    loc,
                    f"run_length_relative_value_{item_index}",
                )
                statements.append(
                    ir.Assign(
                        ir.Expr.getitem(relative_offsets, item_index_var, loc),
                        relative_value,
                        loc,
                    )
                )
                assert relative_sentinel is not None
                projected_relative = self._emit_factory_call(
                    statements,
                    scope=scope,
                    loc=loc,
                    stem=f"run_length_projected_relative_{item_index}",
                    factory=cuda_module.selp,
                    args=[is_valid, relative_value, relative_sentinel],
                    kwargs={},
                )
                statements.append(
                    ir.SetItem(
                        relative_offsets,
                        item_index_var,
                        projected_relative,
                        loc,
                    )
                )
        statements.append(ir.Assign(decoded_items, inst.target, loc))
        return statements


__all__ = ["_RunLengthDecodePlanning"]
