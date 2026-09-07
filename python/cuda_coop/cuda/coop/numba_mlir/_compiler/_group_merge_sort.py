# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Merge Sort IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

from ._group_planner_support import (
    _PAYLOAD_DTYPE_LIKE,
    Any,
    Integral,
    ThreadGroup,
    _builtin_greater,
    _builtin_less,
    inspect,
    ir,
    np,
)


class _MergeSortPlanning:
    def _lower_merge_sort(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        self._reject_extra_root_arguments(operation, bound)
        if group.kind not in {"block", "warp", "threads_within_warp"}:
            raise NotImplementedError(
                f"cuda.coop.numba_mlir.{operation} currently lowers only "
                "physical block, physical-warp, and logical-warp groups"
            )

        descending = self._constant(bound.arguments["descending"])
        if not isinstance(descending, bool):
            raise TypeError(
                f"cuda.coop.numba_mlir.{operation} descending must be a "
                "compile-time bool"
            )
        compare_arg = bound.arguments.get("compare_op")
        if self._is_none(compare_arg):
            compare_op = _builtin_greater if descending else _builtin_less
        else:
            if descending:
                raise ValueError(
                    f"cuda.coop.numba_mlir.{operation} custom compare_op and "
                    "descending=True are mutually exclusive"
                )
            compare_op = compare_arg

        has_valid_items = not self._is_none(bound.arguments["valid_items"])
        has_oob_default = not self._is_none(bound.arguments["oob_default"])
        if has_valid_items != has_oob_default:
            raise ValueError(
                f"cuda.coop.numba_mlir.{operation} valid_items and "
                "oob_default must be provided together"
            )
        if has_valid_items:
            is_static, valid_items = self._try_constant(bound.arguments["valid_items"])
            if is_static:
                if isinstance(valid_items, (bool, np.bool_)) or not isinstance(
                    valid_items, Integral
                ):
                    raise TypeError(
                        f"cuda.coop.numba_mlir.{operation} valid_items must be "
                        "an integer, not bool"
                        if isinstance(valid_items, (bool, np.bool_))
                        else f"cuda.coop.numba_mlir.{operation} valid_items "
                        "must be an integer"
                    )
                items_per_thread = self._array_extent(bound.arguments["keys"])
                if items_per_thread is None and not self._array_operand_state(
                    operation, bound.arguments["keys"]
                ):
                    items_per_thread = 1
                group_size = group.static_size
                if items_per_thread is not None and group_size is not None:
                    maximum = group_size * items_per_thread
                    if not 0 <= int(valid_items) <= maximum:
                        raise ValueError(
                            f"cuda.coop.numba_mlir.{operation} static "
                            f"valid_items must be in [0, {maximum}]"
                        )

        factory, factory_kwargs = self._scope_factory(group, operation)
        factory_kwargs["compare_op"] = compare_op
        if has_valid_items:
            factory_kwargs["valid_items"] = bound.arguments["valid_items"]
            factory_kwargs["oob_default"] = bound.arguments["oob_default"]
        if not self._is_none(bound.arguments["temp_storage"]):
            if group.kind != "block":
                raise NotImplementedError(
                    "cuda.coop.numba_mlir Merge Sort TempStorage is supported "
                    "only for block groups"
                )
            factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]

        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        keys = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_keys",
            value=bound.arguments["keys"],
        )
        if is_common_root and not self._thread_data_operand_state(
            operation, "keys", keys
        ):
            raise TypeError(
                f"cuda.coop.{operation} requires keys to be fixed-size "
                "ThreadData in the portable API; use cuda.coop.numba_mlir for "
                "backend-qualified scalar or local-array payloads"
            )
        keys_payload, keys_are_array = self._boxed_group_operand(
            statements,
            operation=operation,
            value=keys,
            scope=scope,
            loc=loc,
        )
        result_keys = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_keys_result",
            prototype=keys,
            is_array=keys_are_array,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
        )
        self._copy_array_payload(
            statements,
            operation=operation,
            source=keys_payload,
            destination=result_keys,
            scope=scope,
            loc=loc,
            known_items_per_thread=1 if not keys_are_array else None,
        )

        runtime_args = [result_keys]
        result_values = None
        values_are_array = False
        if operation == "merge_sort_pairs":
            values = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_values",
                value=bound.arguments["values"],
            )
            if is_common_root and not self._thread_data_operand_state(
                operation, "values", values
            ):
                raise TypeError(
                    "cuda.coop.merge_sort_pairs requires values to be "
                    "fixed-size ThreadData in the portable API; use "
                    "cuda.coop.numba_mlir for backend-qualified scalar or "
                    "local-array payloads"
                )
            values_payload, values_are_array = self._boxed_group_operand(
                statements,
                operation=operation,
                value=values,
                scope=scope,
                loc=loc,
            )
            if values_are_array != keys_are_array:
                raise TypeError(
                    f"cuda.coop.numba_mlir.{operation} keys and values must "
                    "have the same scalar or ThreadData shape"
                )
            if self._array_extent(values_payload) != self._array_extent(keys_payload):
                raise ValueError(
                    f"cuda.coop.numba_mlir.{operation} keys and values must "
                    "have the same items_per_thread"
                )
            result_values = self._typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_values_result",
                prototype=values,
                is_array=values_are_array,
                dtype_policy=_PAYLOAD_DTYPE_LIKE,
            )
            self._copy_array_payload(
                statements,
                operation=operation,
                source=values_payload,
                destination=result_values,
                scope=scope,
                loc=loc,
                known_items_per_thread=1 if not values_are_array else None,
            )
            runtime_args.append(result_values)

        call_statements = self._rewritten_call(
            inst,
            factory=factory,
            args=runtime_args,
            kwargs=factory_kwargs,
            return_alias=(
                result_keys if result_values is None else (result_keys, result_values)
            ),
            common_root_operation=(operation if is_common_root else None),
        )
        call_statements.pop()
        statements.extend(call_statements)
        keys_result = self._result_value(
            statements,
            payload=result_keys,
            is_array=keys_are_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_keys_result",
        )
        if result_values is None:
            statements.append(ir.Assign(keys_result, inst.target, loc))
            return statements

        values_result = self._result_value(
            statements,
            payload=result_values,
            is_array=values_are_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_values_result",
        )
        statements.append(
            ir.Assign(
                ir.Expr.build_tuple([keys_result, values_result], loc),
                inst.target,
                loc,
            )
        )
        return statements


__all__ = ["_MergeSortPlanning"]
