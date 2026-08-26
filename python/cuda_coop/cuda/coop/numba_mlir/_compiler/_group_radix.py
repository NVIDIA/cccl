# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Radix rank and sort IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

from ._group_planner_support import (
    _PAYLOAD_DTYPE_INT32,
    _PAYLOAD_DTYPE_LIKE,
    Any,
    ThreadGroup,
    _static_bool,
    _static_index,
    inspect,
    ir,
)


class _RadixPlanning:
    def _lower_radix_rank(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        operation = "radix_rank"
        scope_name = "cuda.coop" if is_common_root else "cuda.coop.numba_mlir"
        self._reject_extra_root_arguments(operation, bound)
        if group.kind != "block":
            raise NotImplementedError(
                f"{scope_name}.radix_rank currently lowers only complete "
                "physical block groups"
            )
        if is_common_root and not self._thread_data_operand_state(
            operation, "keys", bound.arguments["keys"]
        ):
            raise TypeError(
                "cuda.coop.radix_rank requires keys to be coop.ThreadData "
                "in the portable API; use cuda.coop.numba_mlir for "
                "backend-qualified payloads"
            )

        begin_bit = _static_index(
            scope_name,
            operation,
            "begin_bit",
            self._constant(bound.arguments["begin_bit"]),
        )
        end_bit = (
            None
            if self._is_none(bound.arguments["end_bit"])
            else _static_index(
                scope_name,
                operation,
                "end_bit",
                self._constant(bound.arguments["end_bit"]),
            )
        )
        radix_bits = (
            None
            if self._is_none(bound.arguments["radix_bits"])
            else _static_index(
                scope_name,
                operation,
                "radix_bits",
                self._constant(bound.arguments["radix_bits"]),
            )
        )
        if begin_bit < 0:
            raise ValueError(f"{scope_name}.radix_rank begin_bit must be non-negative")
        if radix_bits is not None and radix_bits <= 0:
            raise ValueError(f"{scope_name}.radix_rank radix_bits must be positive")
        if end_bit is None:
            end_bit = begin_bit + (4 if radix_bits is None else radix_bits)
        elif radix_bits is not None and end_bit != begin_bit + radix_bits:
            raise ValueError(
                f"{scope_name}.radix_rank radix_bits must match end_bit - begin_bit"
            )
        if end_bit <= begin_bit:
            raise ValueError(
                f"{scope_name}.radix_rank end_bit must be greater than begin_bit"
            )
        if end_bit - begin_bit > 8:
            raise ValueError(f"{scope_name}.radix_rank bit width must be <= 8")
        descending = _static_bool(
            scope_name,
            operation,
            "descending",
            self._constant(bound.arguments["descending"]),
        )

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
        keys_payload, is_array = self._boxed_group_operand(
            statements,
            operation=operation,
            value=keys,
            scope=scope,
            loc=loc,
        )
        ranks_payload = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_result",
            prototype=keys,
            is_array=is_array,
            dtype_policy=_PAYLOAD_DTYPE_INT32,
        )

        factory, factory_kwargs = self._scope_factory(group, operation)
        if is_common_root:
            from .._lowering._radix import _common_radix_rank

            factory = _common_radix_rank
        factory_kwargs.update(
            {
                "begin_bit": begin_bit,
                "end_bit": end_bit,
                "descending": descending,
            }
        )
        prefix = bound.arguments.get("exclusive_digit_prefix")
        if prefix is not None and not self._is_none(prefix):
            if not self._array_operand_state(operation, prefix):
                raise TypeError(
                    "cuda.coop.numba_mlir.radix_rank "
                    "exclusive_digit_prefix must be an explicit array payload"
                )
            prefix_extent = self._array_extent(prefix)
            group_size = group.static_size
            if prefix_extent is not None and group_size is not None:
                expected = max(
                    1, ((1 << (end_bit - begin_bit)) + group_size - 1) // group_size
                )
                if prefix_extent != expected:
                    raise ValueError(
                        "cuda.coop.numba_mlir.radix_rank "
                        "exclusive_digit_prefix must contain "
                        f"{expected} items per thread"
                    )
            factory_kwargs["exclusive_digit_prefix"] = prefix

        call_statements = self._rewritten_call(
            inst,
            factory=factory,
            args=[keys_payload, ranks_payload],
            kwargs=factory_kwargs,
            return_alias=ranks_payload,
        )
        call_statements.pop()
        statements.extend(call_statements)
        result = self._result_value(
            statements,
            payload=ranks_payload,
            is_array=is_array,
            scope=scope,
            loc=loc,
            stem=f"{operation}_result",
        )
        statements.append(ir.Assign(result, inst.target, loc))
        return statements

    def _lower_radix_sort(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        self._reject_extra_root_arguments(operation, bound)
        scope_name = "cuda.coop" if is_common_root else "cuda.coop.numba_mlir"
        if group.kind != "block":
            raise NotImplementedError(
                f"{scope_name}.{operation} currently lowers only complete "
                "physical block groups"
            )
        descending = _static_bool(
            scope_name,
            operation,
            "descending",
            self._constant(bound.arguments["descending"]),
        )
        blocked_to_striped = _static_bool(
            scope_name,
            operation,
            "blocked_to_striped",
            self._constant(bound.arguments.get("blocked_to_striped", False)),
        )
        if is_common_root:
            parameters = (
                ("keys", "values") if operation.endswith("_pairs") else ("keys",)
            )
            for parameter in parameters:
                if not self._thread_data_operand_state(
                    operation, parameter, bound.arguments[parameter]
                ):
                    raise TypeError(
                        f"cuda.coop.{operation} requires {parameter} to be "
                        "coop.ThreadData in the portable API; use cuda.coop.numba_mlir "
                        "for backend-qualified payloads"
                    )

        factory_operation = f"{operation}_descending" if descending else operation
        factory, factory_kwargs = self._scope_factory(group, factory_operation)
        if is_common_root:
            from .._lowering import _radix as _radix_lowering

            factory = getattr(_radix_lowering, f"_common_{operation}")
            factory_kwargs["descending"] = descending
        elif blocked_to_striped:
            factory_kwargs["blocked_to_striped"] = True
        if not self._is_none(bound.arguments["temp_storage"]):
            factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]

        begin_bit = bound.arguments["begin_bit"]
        end_bit = bound.arguments["end_bit"]
        begin_is_static, static_begin = self._try_constant(begin_bit)
        end_is_none = self._is_none(end_bit)
        end_is_static, static_end = (
            (True, None) if end_is_none else self._try_constant(end_bit)
        )
        if begin_is_static:
            static_begin = _static_index(
                scope_name, operation, "begin_bit", static_begin
            )
            if static_begin < 0:
                raise ValueError(
                    f"{scope_name}.{operation} begin_bit must be non-negative"
                )
        if end_is_static and not end_is_none:
            static_end = _static_index(scope_name, operation, "end_bit", static_end)
            if static_end < 1:
                raise ValueError(f"{scope_name}.{operation} end_bit must be positive")
            if begin_is_static and static_end <= static_begin:
                raise ValueError(
                    f"{scope_name}.{operation} end_bit must be greater than begin_bit"
                )
        if not end_is_none:
            factory_kwargs["begin_bit"] = begin_bit
            factory_kwargs["end_bit"] = end_bit
        elif not (begin_is_static and static_begin == 0):
            factory_kwargs["begin_bit"] = begin_bit
            factory_kwargs["end_bit"] = None

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
        if operation == "radix_sort_pairs":
            values = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_values",
                value=bound.arguments["values"],
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
                    f"{scope_name}.{operation} keys and values must have "
                    "the same scalar or ThreadData shape"
                )
            if self._array_extent(values_payload) != self._array_extent(keys_payload):
                raise ValueError(
                    f"{scope_name}.{operation} keys and values must have "
                    "the same items_per_thread"
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


__all__ = ["_RadixPlanning"]
