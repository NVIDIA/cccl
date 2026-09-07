# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar BlockReduce family planning after block hierarchy resolution."""

from __future__ import annotations

from cuda.coop._core._bindings import ArgumentBinding
from cuda.coop._core.block.reduce import BlockReduceOperator
from cuda.coop._core.group import (
    GroupReduceSemantics,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop._core.launch import LaunchFactOrigin, LaunchFacts
from cuda.coop._core.thread_group import this_block

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
        if not self._is_descriptor(bound.arguments["group"]):
            raise self.error_type(
                "cuda.coop block reduction group must come from this_block()"
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
                this_block(),
                semantics,
                source="numba-cuda-mlir hierarchy planner",
            ),
            launch,
        ).require_supported()
        implementation = plan.implementation
        assert implementation is not None

        from .._lowering._reduce import block_reduce_builtin, sum

        if implementation.binary_op is BlockReduceOperator.SUM:
            factory = sum
            factory_kwargs = {
                "threads_per_block": implementation.block_dim,
                "algorithm": implementation.algorithm.value,
            }
        else:
            factory = block_reduce_builtin
            factory_kwargs = {
                "threads_per_block": implementation.block_dim,
                "binary_op": implementation.binary_op.value,
                "algorithm": implementation.algorithm.value,
            }
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
