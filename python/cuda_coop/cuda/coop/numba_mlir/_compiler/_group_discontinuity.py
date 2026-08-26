# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Discontinuity IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

from ._group_planner_support import (
    _PAYLOAD_DTYPE_INT32,
    Any,
    ThreadGroup,
    _builtin_not_equal,
    inspect,
    ir,
    types,
)


class _DiscontinuityPlanning:
    def _lower_discontinuity(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        operation = "discontinuity"
        self._reject_extra_root_arguments(operation, bound)
        if group.kind != "block":
            raise NotImplementedError(
                "cuda.coop.numba_mlir.discontinuity currently lowers only "
                "complete physical block groups"
            )

        flag_argument = bound.arguments.get("flag_op")
        if self._is_none(flag_argument):
            flag_op = _builtin_not_equal
        else:
            flag_op = self._constant(flag_argument)
            if not callable(flag_op):
                raise TypeError(
                    "cuda.coop.numba_mlir.discontinuity flag_op must be a "
                    "device callable"
                )
            if is_common_root:
                raise ValueError(
                    "cuda.coop.discontinuity uses built-in inequality in the "
                    "portable API"
                )

        from cuda.coop._core.block import BlockDiscontinuityMode

        from .._lowering._discontinuity import BlockDiscontinuityType

        try:
            mode = BlockDiscontinuityMode(self._constant(bound.arguments["mode"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "cuda.coop.numba_mlir.discontinuity mode must be "
                "'heads', 'tails', or 'heads_and_tails'"
            ) from exc
        discontinuity_type = {
            BlockDiscontinuityMode.HEADS: BlockDiscontinuityType.HEADS,
            BlockDiscontinuityMode.TAILS: BlockDiscontinuityType.TAILS,
            BlockDiscontinuityMode.HEADS_AND_TAILS: (
                BlockDiscontinuityType.HEADS_AND_TAILS
            ),
        }[mode]

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
                "cuda.coop.discontinuity requires a fixed-size ThreadData "
                "payload in the portable API; use cuda.coop.numba_mlir for "
                "qualified scalar or local-array support"
            )
        input_payload, is_array = self._boxed_group_operand(
            statements,
            operation=operation,
            value=value,
            scope=scope,
            loc=loc,
        )
        head_payload = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_head_result",
            prototype=value,
            is_array=is_array,
            dtype_policy=_PAYLOAD_DTYPE_INT32,
        )
        tail_payload = None
        if mode is BlockDiscontinuityMode.HEADS_AND_TAILS:
            tail_payload = self._typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_tail_result",
                prototype=value,
                is_array=is_array,
                dtype_policy=_PAYLOAD_DTYPE_INT32,
            )

        factory, factory_kwargs = self._scope_factory(group, operation)
        factory_kwargs.update(
            {
                "block_discontinuity_type": discontinuity_type,
                "flag_op": flag_op,
                "flag_dtype": types.int32,
            }
        )
        runtime_args = [input_payload, head_payload]
        return_payload: ir.Var | tuple[ir.Var, ...] = head_payload
        if tail_payload is not None:
            runtime_args.append(tail_payload)
            return_payload = (head_payload, tail_payload)
        predecessor = bound.arguments["tile_predecessor_item"]
        successor = bound.arguments["tile_successor_item"]
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
            return_alias=return_payload,
            common_root_operation=operation if is_common_root else None,
        )
        call_statements.pop()
        statements.extend(call_statements)
        head_result = self._result_value(
            statements,
            payload=head_payload,
            is_array=is_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_head_result",
        )
        if tail_payload is None:
            statements.append(ir.Assign(head_result, inst.target, loc))
            return statements
        tail_result = self._result_value(
            statements,
            payload=tail_payload,
            is_array=is_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_tail_result",
        )
        statements.append(
            ir.Assign(
                ir.Expr.build_tuple([head_result, tail_result], loc),
                inst.target,
                loc,
            )
        )
        return statements


__all__ = ["_DiscontinuityPlanning"]
