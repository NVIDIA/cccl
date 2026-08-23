# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TopK IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

from ._group_planner_support import (
    _PAYLOAD_DTYPE_LIKE,
    Any,
    GroupRewriteError,
    ThreadGroup,
    inspect,
    ir,
    np,
    operator,
)


class _TopKPlanning:
    def _lower_topk(
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
        assert group.hierarchy is not None
        block_dim = group.hierarchy.block_dim
        if block_dim is None or block_dim[1:] != (1, 1):
            raise ValueError(
                f"{scope_name}.{operation} requires a one-dimensional block"
            )
        if group.static_size is None:
            raise GroupRewriteError(
                f"{scope_name}.{operation} requires a static block size"
            )
        if group.static_size > 1024:
            raise ValueError(
                f"{scope_name}.{operation} block thread count must be <= 1024"
            )

        parameters = ("keys", "values") if operation.endswith("_pairs") else ("keys",)
        if is_common_root:
            for parameter in parameters:
                is_thread_data = self._is_array_value(
                    bound.arguments[parameter],
                    thread_data_only=True,
                )
                if is_thread_data is None:
                    raise GroupRewriteError(
                        f"{scope_name}.{operation} could not resolve {parameter} "
                        "payload provenance"
                    )
                if not is_thread_data:
                    raise TypeError(
                        f"{scope_name}.{operation} requires {parameter} to be "
                        "coop.ThreadData in the portable API; use "
                        "cuda.coop.numba_mlir for backend-qualified payloads"
                    )

        items_per_thread = self._array_extent(bound.arguments["keys"])
        if items_per_thread is None:
            raise GroupRewriteError(
                f"{scope_name}.{operation} could not infer a static "
                "items_per_thread extent"
            )
        if items_per_thread <= 0:
            raise ValueError(
                f"{scope_name}.{operation} keys.items_per_thread must be positive"
            )
        if operation.endswith("_pairs"):
            values_extent = self._array_extent(bound.arguments["values"])
            if values_extent != items_per_thread:
                raise ValueError(
                    f"{scope_name}.{operation} keys and values must have the "
                    "same items_per_thread"
                )

        def static_int(name: str, value: Any) -> int | None:
            is_static, static_value = self._try_constant(value)
            if not is_static:
                return None
            if isinstance(static_value, (bool, np.bool_)):
                raise TypeError(
                    f"{scope_name}.{operation} {name} must be an int-like scalar"
                )
            try:
                normalized = operator.index(static_value)
            except TypeError as exc:
                raise TypeError(
                    f"{scope_name}.{operation} {name} must be an int-like scalar"
                ) from exc
            if isinstance(normalized, bool):
                raise TypeError(
                    f"{scope_name}.{operation} {name} must be an int-like scalar"
                )
            return int(normalized)

        static_k = static_int("k", bound.arguments["k"])
        if static_k is not None and static_k <= 0:
            raise ValueError(f"{scope_name}.{operation} k must be positive")
        tile_size = group.static_size * items_per_thread
        if self._is_none(bound.arguments["valid_items"]):
            static_valid_items = tile_size
        else:
            static_valid_items = static_int(
                "valid_items",
                bound.arguments["valid_items"],
            )
            if static_valid_items is not None and not (
                1 <= static_valid_items <= tile_size
            ):
                raise ValueError(
                    f"{scope_name}.{operation} valid_items must be in [1, {tile_size}]"
                )
        if (
            static_k is not None
            and static_valid_items is not None
            and static_k > static_valid_items
        ):
            raise ValueError(f"{scope_name}.{operation} k must be <= valid_items")

        begin_bit = bound.arguments["begin_bit"]
        end_bit = bound.arguments["end_bit"]
        begin_is_static, static_begin_value = self._try_constant(begin_bit)
        static_begin = static_int("begin_bit", begin_bit)
        if static_begin is not None and static_begin < 0:
            raise ValueError(f"{scope_name}.{operation} begin_bit must be non-negative")
        if self._is_none(end_bit):
            static_end = None
        else:
            static_end = static_int("end_bit", end_bit)
            if static_end is not None and static_end < 1:
                raise ValueError(f"{scope_name}.{operation} end_bit must be positive")
            if (
                static_begin is not None
                and static_end is not None
                and static_end <= static_begin
            ):
                raise ValueError(
                    f"{scope_name}.{operation} end_bit must exceed begin_bit"
                )

        from .._lowering import _topk as _topk_lowering

        factory = getattr(_topk_lowering, operation)
        factory_kwargs = {"threads_per_block": block_dim}
        if is_common_root:
            factory = getattr(_topk_lowering, f"_common_{operation}")
        elif self._is_none(end_bit) and not (
            begin_is_static and static_begin_value == 0
        ):
            factory = getattr(_topk_lowering, f"_qualified_group_{operation}")
        if not self._is_none(bound.arguments["valid_items"]):
            factory_kwargs["num_valid"] = bound.arguments["valid_items"]
        if self._is_none(end_bit):
            if not (begin_is_static and static_begin_value == 0):
                factory_kwargs["begin_bit"] = begin_bit
        else:
            factory_kwargs["begin_bit"] = begin_bit
            factory_kwargs["end_bit"] = end_bit
        if not self._is_none(bound.arguments["temp_storage"]):
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
        if not self._array_operand_state(operation, keys):
            raise TypeError(
                f"{scope_name}.{operation} requires a fixed-size key payload"
            )
        result_keys = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem=f"{operation}_keys_result",
            prototype=keys,
            is_array=True,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
        )
        self._copy_array_payload(
            statements,
            operation=operation,
            source=keys,
            destination=result_keys,
            scope=scope,
            loc=loc,
            known_items_per_thread=items_per_thread,
        )

        runtime_args = [result_keys]
        result_values = None
        if operation.endswith("_pairs"):
            values = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_values",
                value=bound.arguments["values"],
            )
            if not self._array_operand_state(operation, values):
                raise TypeError(
                    f"{scope_name}.{operation} requires a fixed-size value payload"
                )
            result_values = self._typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem=f"{operation}_values_result",
                prototype=values,
                is_array=True,
                dtype_policy=_PAYLOAD_DTYPE_LIKE,
            )
            self._copy_array_payload(
                statements,
                operation=operation,
                source=values,
                destination=result_values,
                scope=scope,
                loc=loc,
                known_items_per_thread=items_per_thread,
            )
            runtime_args.append(result_values)
        runtime_args.append(bound.arguments["k"])

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
        if result_values is None:
            statements.append(ir.Assign(result_keys, inst.target, loc))
        else:
            statements.append(
                ir.Assign(
                    ir.Expr.build_tuple([result_keys, result_values], loc),
                    inst.target,
                    loc,
                )
            )
        return statements


__all__ = ["_TopKPlanning"]
