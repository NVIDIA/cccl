# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar group-reduction planning after hierarchy resolution."""

from __future__ import annotations

from cuda.coop._core._bindings import ArgumentBinding
from cuda.coop._core.block.reduce import BlockReduceOperator
from cuda.coop._core.group import (
    GroupLoweringTarget,
    GroupReduceSemantics,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.launch import LaunchFactOrigin, LaunchFacts

from ._group_planner_support import ir


class _ReducePlanning:
    def _lower_reduce(
        self,
        inst: ir.Assign,
        call: ir.Expr,
        function,
        operation: str,
    ) -> None:
        bound = self._bind(function, call)
        group = self._descriptor(bound.arguments["group"])
        if group is None or group.kind not in {"block", "warp"}:
            raise self.error_type(
                "cuda.coop reduction group must come from this_block() or this_warp()"
            )

        algorithm = self._constant(bound.arguments["algorithm"], name="algorithm")
        binary_op = "sum"
        if operation == "reduce":
            binary_op = self._constant(bound.arguments["binary_op"], name="binary_op")

        valid_items = bound.arguments["valid_items"]
        has_valid_items = not self._is_none(valid_items)
        if has_valid_items:
            is_static, static_valid_items = self._try_constant(
                valid_items, name="valid_items"
            )
            valid_binding = (
                ArgumentBinding.static(static_valid_items)
                if is_static
                else ArgumentBinding.runtime()
            )
        else:
            valid_binding = ArgumentBinding.omitted()

        semantics = GroupReduceSemantics(
            dtype=None,
            operation=operation,
            binary_op=binary_op,
            algorithm=algorithm,
            valid_items=valid_binding,
        )
        launch = LaunchFacts(
            exact_block_dim=self.block_dim,
            provenance=LaunchFactOrigin(
                fact="exact_block_dim",
                source="Numba-CUDA-MLIR launch configuration",
                verified=True,
            ),
        )
        plan = plan_group_primitive(
            make_group_primitive_call(
                group,
                semantics,
                source="numba-cuda-mlir hierarchy planner",
            ),
            launch,
        ).require_supported()
        implementation = plan.implementation
        assert implementation is not None

        from .._lowering._reduce import (
            block_reduce_builtin,
            sum,
            warp_reduce_builtin,
            warp_sum,
        )

        if plan.target is GroupLoweringTarget.CUB_BLOCK:
            factory = (
                sum
                if implementation.binary_op is BlockReduceOperator.SUM
                else block_reduce_builtin
            )
            factory_kwargs = {
                "threads_per_block": implementation.block_dim,
                "algorithm": implementation.algorithm.value,
            }
            if implementation.binary_op is not BlockReduceOperator.SUM:
                factory_kwargs["binary_op"] = implementation.binary_op.value
        elif plan.target is GroupLoweringTarget.CUB_WARP:
            factory = (
                warp_sum
                if implementation.binary_op is BlockReduceOperator.SUM
                else warp_reduce_builtin
            )
            factory_kwargs = {
                "threads_per_block": implementation.block_dim,
            }
            if implementation.binary_op is not BlockReduceOperator.SUM:
                factory_kwargs["binary_op"] = implementation.binary_op.value
        else:
            raise self.error_type(
                f"unsupported cuda.coop lowering target {plan.target.value!r}"
            )
        if has_valid_items:
            factory_kwargs["num_valid"] = valid_items
        self.replacements[inst] = self._rewritten_call(
            inst,
            factory=factory,
            args=[bound.arguments["value"]],
            kwargs=factory_kwargs,
        )
        self.dead_func_names.add(call.func.name)


__all__ = ["_ReducePlanning"]
