# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate shared Load destinations and select each physical thread's row."""

import math
import operator
from numbers import Integral

from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda.shared import array as shared_array

from ._group_planner_support import GroupRewriteError, ir


def _shared_extent(context, value, seen=None):
    seen = set() if seen is None else seen
    if not isinstance(value, ir.Var) or value.name in seen:
        raise TypeError(
            "cuda.coop.load output with items_per_thread must be a "
            "one-dimensional shared array"
        )
    seen.add(value.name)
    definition = context._definition(value)
    if isinstance(definition, ir.Var):
        return _shared_extent(context, definition, seen)
    if isinstance(definition, ir.Expr):
        if definition.op == "cast":
            return _shared_extent(context, definition.value, seen)
        if definition.op in {"getitem", "static_getitem"}:
            extent = _shared_extent(context, definition.value, seen)
            resolved, selected = context.try_constant(definition.index)
            if not resolved:
                index = context._definition(definition.index)
                if (
                    isinstance(index, ir.Expr)
                    and index.op == "call"
                    and context._callable(index.func) is slice
                ):
                    selected = slice(
                        *(context.constant(arg) for arg in index.args)
                    )
                else:
                    selected = None
            if not isinstance(selected, slice) or selected.step not in (
                None,
                1,
            ):
                raise TypeError(
                    "cuda.coop.load shared output must be a contiguous "
                    "one-dimensional array"
                )
            return len(range(extent)[selected])
        if (
            definition.op == "call"
            and context._callable(definition.func) is shared_array
        ):
            shape = context.constant(
                definition.args[0]
                if definition.args
                else dict(definition.kws)["shape"]
            )
            if isinstance(shape, tuple) and len(shape) == 1:
                shape = shape[0]
            if (
                isinstance(shape, Integral)
                and not isinstance(shape, bool)
                and int(shape) > 0
            ):
                return int(shape)
            raise TypeError(
                "cuda.coop.load shared output requires a positive, "
                "one-dimensional static extent"
            )
    raise TypeError(
        "cuda.coop.load output with items_per_thread must be a "
        "one-dimensional shared array"
    )


def validate_shared_load_output(context, output, count, block_dim):
    """Prove every physical thread has a complete, disjoint output row."""

    count = context.constant(count)
    if not isinstance(count, Integral) or isinstance(count, bool) or count <= 0:
        raise ValueError(
            "cuda.coop.load items_per_thread must be a positive "
            "compile-time integer"
        )
    if block_dim is None:
        raise GroupRewriteError(
            "cuda.coop.load shared output requires exact block dimensions"
        )
    extent = _shared_extent(context, output)
    required = math.prod(block_dim) * int(count)
    if extent < required:
        raise ValueError(
            f"cuda.coop.load shared output needs at least {required} "
            f"elements for this block; destination has {extent}"
        )
    return int(count)


def shared_load_thread_row(context, statements, inst, output, count, block_dim):
    """Pass a shared row directly to CUB's existing T (&items)[N] output."""

    scope, loc = inst.target.scope, inst.loc

    def variable(stem):
        return context.new_var(scope, loc, f"shared_load_{stem}")

    def value(value, stem):
        return context.value_var(
            statements, scope=scope, loc=loc, stem=stem, value=value
        )

    def assign(expression, stem):
        result = variable(stem)
        statements.append(ir.Assign(expression, result, loc))
        return result

    def binary(function, left, right, stem):
        return assign(ir.Expr.binop(function, left, right, loc), stem)

    thread = assign(
        ir.Expr.getattr(value(cuda, "cuda"), "threadIdx", loc), "thread"
    )
    rank = assign(ir.Expr.getattr(thread, "x", loc), "rank_x")
    if block_dim[1] > 1 or block_dim[2] > 1:
        y = assign(ir.Expr.getattr(thread, "y", loc), "rank_y")
        z = assign(ir.Expr.getattr(thread, "z", loc), "rank_z")
        yz = binary(operator.mul, value(block_dim[1], "block_y"), z, "yz")
        yz = binary(operator.add, y, yz, "y")
        yz = binary(operator.mul, value(block_dim[0], "block_x"), yz, "xyz")
        rank = binary(operator.add, rank, yz, "rank")
    items = value(count, "shared_load_items")
    start = binary(operator.mul, rank, items, "start")
    stop = binary(operator.add, start, items, "stop")
    selected = assign(
        ir.Expr.call(value(slice, "slice"), [start, stop], (), loc), "slice"
    )
    return assign(ir.Expr.getitem(output, selected, loc), "row")
