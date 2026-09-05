# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Histogram IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

from ._group_planner_support import (
    _PAYLOAD_DTYPE_LIKE,
    Any,
    ThreadGroup,
    _histogram_provider_counter_dtype,
    inspect,
    ir,
    operator,
    types,
)


class _HistogramPlanning:
    def _lower_histogram(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        operation = "histogram"
        self._reject_extra_root_arguments(operation, bound)
        if group.kind != "block":
            raise NotImplementedError(
                "cuda.coop.numba_mlir.histogram currently lowers only "
                "complete physical block groups"
            )

        from cuda.coop._core.block import (
            normalize_block_histogram_positive_int,
            validate_block_histogram_output_capacity,
        )

        bins = normalize_block_histogram_positive_int(
            "bins",
            self._constant(bound.arguments["bins"]),
            scope="cuda.coop.numba_mlir.histogram",
        )
        bins_per_thread = normalize_block_histogram_positive_int(
            "bins_per_thread",
            self._constant(bound.arguments["bins_per_thread"]),
            scope="cuda.coop.numba_mlir.histogram",
        )
        group_size = group.static_size
        assert group_size is not None
        validate_block_histogram_output_capacity(
            bins=bins,
            bins_per_thread=bins_per_thread,
            block_threads=group_size,
            scope="cuda.coop.numba_mlir.histogram",
        )

        if is_common_root and not self._thread_data_operand_state(
            operation,
            "samples",
            bound.arguments["samples"],
        ):
            raise TypeError(
                "cuda.coop.histogram requires samples to be coop.ThreadData "
                "in the portable API; use cuda.coop.numba_mlir for "
                "backend-qualified local-array payloads"
            )

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        samples = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_samples",
            value=bound.arguments["samples"],
        )
        if not self._array_operand_state(operation, samples):
            raise TypeError(
                "cuda.coop.numba_mlir.histogram requires a fixed-size "
                "ThreadData or local-array samples payload"
            )
        provider_samples = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_provider_samples",
            prototype=samples,
            is_array=True,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
        )
        self._copy_array_payload(
            statements,
            operation=operation,
            source=samples,
            destination=provider_samples,
            scope=scope,
            loc=loc,
        )

        counter_dtype = bound.arguments["counter_dtype"]
        if self._is_none(counter_dtype):
            counter_dtype = types.int32
        else:
            from ._parameters import normalize_dtype_param

            counter_dtype = self._constant(counter_dtype)
            counter_dtype = (
                types.int32
                if counter_dtype is int
                else normalize_dtype_param(counter_dtype)
            )
        provider_counter_dtype = _histogram_provider_counter_dtype(counter_dtype)
        histogram = self._emit_shared_array(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_counters",
            items=bins,
            dtype=provider_counter_dtype,
        )
        result = self._thread_data_payload(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_result",
            items_per_thread=bins_per_thread,
            dtype=counter_dtype,
        )

        from .._lowering._histogram import _group_histogram

        assert group.hierarchy is not None
        self._emit_factory_call(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_fused",
            factory=_group_histogram,
            args=[provider_samples, histogram],
            kwargs={
                "threads_per_block": group.hierarchy.block_dim,
                "bins": bins,
                "algorithm": bound.arguments["algorithm"],
                **({"_common_root_operation": operation} if is_common_root else {}),
            },
        )
        rank = self._emit_group_method_call(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_rank",
            group=group,
            operation="rank",
        )
        bins_var = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="histogram_bins",
            value=bins,
        )
        for item_index in range(bins_per_thread):
            offset = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"histogram_offset_{item_index}",
                value=item_index * group_size,
            )
            striped_index = self._new_var(
                scope,
                loc,
                f"histogram_striped_index_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.binop(operator.add, rank, offset, loc),
                    striped_index,
                    loc,
                )
            )
            safe_index = self._new_var(
                scope,
                loc,
                f"histogram_safe_index_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.binop(operator.mod, striped_index, bins_var, loc),
                    safe_index,
                    loc,
                )
            )
            counter = self._new_var(
                scope,
                loc,
                f"histogram_counter_{item_index}",
            )
            statements.append(
                ir.Assign(ir.Expr.getitem(histogram, safe_index, loc), counter, loc)
            )
            is_valid = self._new_var(
                scope,
                loc,
                f"histogram_counter_valid_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.binop(operator.lt, striped_index, bins_var, loc),
                    is_valid,
                    loc,
                )
            )
            projected = self._new_var(
                scope,
                loc,
                f"histogram_projected_counter_{item_index}",
            )
            statements.append(
                ir.Assign(
                    ir.Expr.binop(operator.mul, counter, is_valid, loc),
                    projected,
                    loc,
                )
            )
            output_index = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"histogram_output_index_{item_index}",
                value=item_index,
            )
            statements.append(ir.SetItem(result, output_index, projected, loc))
        statements.append(ir.Assign(result, inst.target, loc))
        return statements

    def _emit_group_method_call(
        self,
        statements: list[Any],
        *,
        scope: Any,
        loc: ir.Loc,
        stem: str,
        group: ThreadGroup,
        operation: str,
    ) -> ir.Var:
        """Emit one already-resolved group-method invocable call."""

        from .._lowering._thread_group import make_group_method_invocable

        invocable = make_group_method_invocable(
            group=group,
            operation=operation,
            dtype=None,
            level="thread",
            compile_context=self._provider_compile_context(),
        )
        return self._emit_factory_call(
            statements,
            scope=scope,
            loc=loc,
            stem=stem,
            factory=invocable,
            args=[],
            kwargs={},
        )


__all__ = ["_HistogramPlanning"]
