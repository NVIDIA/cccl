# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Adjacent-difference IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

from ._group_planner_support import (
    _PAYLOAD_DTYPE_LIKE,
    Any,
    GroupRewriteError,
    Integral,
    ThreadGroup,
    _builtin_subtract,
    inspect,
    ir,
)


class _AdjacentDifferencePlanning:
    def _lower_adjacent_difference(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        operation = "adjacent_difference"
        self._reject_extra_root_arguments(operation, bound)
        if group.kind != "block":
            raise NotImplementedError(
                "cuda.coop.numba_mlir.adjacent_difference currently lowers "
                "only complete physical block groups"
            )

        difference_argument = bound.arguments.get("difference_op")
        if self._is_none(difference_argument):
            difference_op = _builtin_subtract
        else:
            difference_op = self._constant(difference_argument)
            if not callable(difference_op):
                raise TypeError(
                    "cuda.coop.numba_mlir.adjacent_difference difference_op "
                    "must be a device callable"
                )
            if is_common_root:
                raise ValueError(
                    "cuda.coop.adjacent_difference uses built-in subtraction "
                    "in the portable API"
                )

        from cuda.coop._core.block import BlockAdjacentDifferenceDirection

        from .._lowering._adjacent_difference import BlockAdjacentDifferenceType

        try:
            direction = BlockAdjacentDifferenceDirection(
                self._constant(bound.arguments["direction"])
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cuda.coop.numba_mlir.adjacent_difference direction must be "
                "'left' or 'right'"
            ) from exc
        adjacent_type = {
            BlockAdjacentDifferenceDirection.LEFT: (
                BlockAdjacentDifferenceType.SubtractLeft
            ),
            BlockAdjacentDifferenceDirection.RIGHT: (
                BlockAdjacentDifferenceType.SubtractRight
            ),
        }[direction]

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        value = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_value",
            value=bound.arguments["value"],
        )
        if is_common_root and not self._thread_data_operand_state(
            operation,
            "value",
            value,
        ):
            raise TypeError(
                "cuda.coop.adjacent_difference requires a fixed-size "
                "ThreadData payload in the portable API; use "
                "cuda.coop.numba_mlir for qualified scalar or local-array "
                "support"
            )
        input_payload, is_array = self._boxed_group_operand(
            statements,
            operation=operation,
            value=value,
            scope=scope,
            loc=loc,
        )
        items_per_thread = self._array_extent(value) if is_array else 1
        if items_per_thread is None:
            raise GroupRewriteError(
                "cuda.coop.numba_mlir.adjacent_difference could not infer a "
                "static items_per_thread extent"
            )
        result_payload = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_result",
            prototype=value,
            is_array=is_array,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
        )

        factory, factory_kwargs = self._scope_factory(group, operation)
        factory_kwargs.update(
            {
                "block_adjacent_difference_type": adjacent_type,
                "difference_op": difference_op,
            }
        )
        runtime_args = [input_payload, result_payload]
        valid_items = bound.arguments["valid_items"]
        predecessor = bound.arguments["tile_predecessor_item"]
        successor = bound.arguments["tile_successor_item"]
        if not self._is_none(valid_items):
            is_static, static_valid_items = self._try_constant(valid_items)
            if is_static:
                if isinstance(static_valid_items, bool) or not isinstance(
                    static_valid_items, Integral
                ):
                    raise TypeError(
                        "cuda.coop.numba_mlir.adjacent_difference valid_items "
                        "must be an integer, not bool"
                    )
                assert group.static_size is not None
                tile_size = group.static_size * items_per_thread
                if not 0 <= int(static_valid_items) <= tile_size:
                    raise ValueError(
                        "cuda.coop.numba_mlir.adjacent_difference static "
                        f"valid_items must be between 0 and tile size {tile_size}"
                    )
            runtime_args.append(valid_items)
            factory_kwargs["valid_items"] = True
        if not self._is_none(predecessor):
            runtime_args.append(predecessor)
            factory_kwargs["tile_predecessor_item"] = True
        if not self._is_none(successor):
            runtime_args.append(successor)
            factory_kwargs["tile_successor_item"] = True
        if not self._is_none(bound.arguments["temp_storage"]):
            factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]

        call_statements = self._rewritten_call(
            inst,
            factory=factory,
            args=runtime_args,
            kwargs=factory_kwargs,
            return_alias=result_payload,
            common_root_operation=operation if is_common_root else None,
        )
        call_statements.pop()
        statements.extend(call_statements)
        result = self._result_value(
            statements,
            payload=result_payload,
            is_array=is_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_result",
        )
        statements.append(ir.Assign(result, inst.target, loc))
        return statements


__all__ = ["_AdjacentDifferencePlanning"]
