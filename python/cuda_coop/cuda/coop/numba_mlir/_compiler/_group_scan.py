# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scan IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

from ._group_planner_support import (
    _PAYLOAD_DTYPE_LIKE,
    Any,
    GroupRewriteError,
    Integral,
    ScanOp,
    ThreadGroup,
    inspect,
    ir,
)


class _ScanPlanning:
    @staticmethod
    def _reject_extra_root_arguments(
        operation: str, bound: inspect.BoundArguments
    ) -> None:
        if bound.arguments.get("args"):
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} accepts no extra positional arguments"
            )
        if bound.arguments.get("kwargs"):
            names = ", ".join(sorted(bound.arguments["kwargs"]))
            raise GroupRewriteError(
                f"cuda.coop.numba_mlir.{operation} got unexpected keyword(s): {names}"
            )

    def _validate_scan_aggregate(
        self,
        bound: inspect.BoundArguments,
    ) -> None:
        aggregate = bound.arguments["aggregate_output"]
        if self._is_none(aggregate):
            return
        if not self._array_operand_state("scan", aggregate):
            raise TypeError(
                "cuda.coop.numba_mlir.scan aggregate_output must be a "
                "single-item ThreadData or local array"
            )
        extent = self._array_extent(aggregate)
        if extent != 1:
            raise ValueError(
                "cuda.coop.numba_mlir.scan aggregate_output must contain "
                "exactly one item"
            )

    def _lower_scan(
        self,
        inst: ir.Assign,
        *,
        operation: str,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        self._reject_extra_root_arguments("scan", bound)
        mode = self._constant(bound.arguments["mode"])
        if mode not in {"exclusive", "inclusive"}:
            raise ValueError(
                "cuda.coop.numba_mlir.scan mode must be 'exclusive' or 'inclusive'"
            )
        if mode == "inclusive" and not self._is_none(bound.arguments["initial_value"]):
            raise ValueError(
                "cuda.coop.numba_mlir.scan initial_value is not supported for "
                "inclusive scans"
            )
        self._validate_scan_aggregate(bound)

        prefix_op = bound.arguments["prefix_op"]
        legacy_prefix_op = bound.arguments["block_prefix_callback_op"]
        has_prefix_op = not self._is_none(prefix_op)
        has_legacy_prefix_op = not self._is_none(legacy_prefix_op)
        if has_prefix_op and has_legacy_prefix_op:
            raise ValueError(
                "cuda.coop.numba_mlir.scan prefix_op and "
                "block_prefix_callback_op are mutually exclusive"
            )
        prefix_ref = prefix_op if has_prefix_op else legacy_prefix_op
        prefix_callback = None
        if not self._is_none(prefix_ref):
            prefix_callback = self._constant(prefix_ref)

        prefix_state = bound.arguments["prefix_state"]
        has_prefix_state = not self._is_none(prefix_state)
        if group.kind != "block" and (prefix_callback is not None or has_prefix_state):
            raise NotImplementedError(
                "cuda.coop.numba_mlir.scan prefix callbacks apply only to block groups"
            )

        from .._stateful_function import StatefulFunction

        stateful_prefix = isinstance(prefix_callback, StatefulFunction)
        if prefix_callback is None and has_prefix_state:
            raise ValueError(
                "cuda.coop.numba_mlir.scan prefix_state requires a prefix callback"
            )
        if stateful_prefix and not has_prefix_state:
            raise ValueError(
                "cuda.coop.numba_mlir.scan StatefulFunction prefix callbacks "
                "require a third positional prefix_state argument"
            )
        if prefix_callback is not None and not stateful_prefix and has_prefix_state:
            raise ValueError(
                "cuda.coop.numba_mlir.scan stateless prefix callbacks do not "
                "accept prefix_state"
            )
        if prefix_callback is not None and not self._is_none(
            bound.arguments["initial_value"]
        ):
            raise ValueError(
                "cuda.coop.numba_mlir.scan initial_value and prefix callbacks "
                "are mutually exclusive"
            )
        if prefix_callback is not None and not self._is_none(
            bound.arguments["aggregate_output"]
        ):
            raise ValueError(
                "cuda.coop.numba_mlir.scan aggregate_output and prefix callbacks "
                "are mutually exclusive"
            )

        if group.kind == "block":
            if not self._is_none(bound.arguments["valid_items"]):
                raise NotImplementedError(
                    "cuda.coop.numba_mlir.scan valid_items applies to physical "
                    "and logical warp groups, not block groups"
                )
            factory, factory_kwargs = self._scope_factory(group, "scan")
            factory_kwargs.update(
                {
                    "mode": mode,
                    "scan_op": (
                        "+"
                        if self._is_none(bound.arguments["scan_op"])
                        else bound.arguments["scan_op"]
                    ),
                }
            )
            if not self._is_none(bound.arguments["initial_value"]):
                factory_kwargs["initial_value"] = bound.arguments["initial_value"]
            if not self._is_none(bound.arguments["algorithm"]):
                factory_kwargs["algorithm"] = bound.arguments["algorithm"]
            if not self._is_none(bound.arguments["temp_storage"]):
                factory_kwargs["temp_storage"] = bound.arguments["temp_storage"]
            if not self._is_none(bound.arguments["aggregate_output"]):
                factory_kwargs["block_aggregate"] = bound.arguments["aggregate_output"]
            if has_prefix_op:
                factory_kwargs["prefix_op"] = prefix_op
            elif has_legacy_prefix_op:
                factory_kwargs["block_prefix_callback_op"] = legacy_prefix_op

            if stateful_prefix:
                if not self._array_operand_state("scan", prefix_state):
                    raise TypeError(
                        "cuda.coop.numba_mlir.scan prefix_state must be a "
                        "one-item ThreadData or local array"
                    )
                if self._array_extent(prefix_state) != 1:
                    raise ValueError(
                        "cuda.coop.numba_mlir.scan prefix_state must contain "
                        "exactly one item"
                    )

            statements: list[Any] = []
            scope = inst.target.scope
            loc = inst.loc
            value = self._value_var(
                statements,
                scope=scope,
                loc=loc,
                stem="scan_value",
                value=bound.arguments["value"],
            )
            if is_common_root and self._array_operand_state("scan", value):
                if not self._thread_data_operand_state(operation, "value", value):
                    raise TypeError(
                        f"cuda.coop.{operation} accepts only a scalar or fixed-size "
                        "ThreadData value payload in the portable API; use "
                        "cuda.coop.numba_mlir for backend-qualified local arrays"
                    )
            input_payload, is_array = self._boxed_group_operand(
                statements,
                operation="scan",
                value=value,
                scope=scope,
                loc=loc,
            )
            result_payload = self._typed_payload_like(
                statements,
                scope=scope,
                loc=loc,
                stem="scan_result",
                prototype=value,
                is_array=is_array,
                dtype_policy=_PAYLOAD_DTYPE_LIKE,
            )
            runtime_args = [input_payload, result_payload]
            if stateful_prefix:
                runtime_args.append(prefix_state)
            call_statements = self._rewritten_call(
                inst,
                factory=factory,
                args=runtime_args,
                kwargs=factory_kwargs,
                return_alias=result_payload,
                common_root_operation=(operation if is_common_root else None),
            )
            call_statements.pop()
            statements.extend(call_statements)
            result = self._result_value(
                statements,
                payload=result_payload,
                is_array=is_array,
                scope=scope,
                loc=loc,
                stem="scan_result",
            )
            statements.append(ir.Assign(result, inst.target, loc))
            return statements

        if group.kind not in {"warp", "threads_within_warp"}:
            raise NotImplementedError(
                "cuda.coop.numba_mlir.scan currently lowers only block, "
                "physical-warp, and logical-warp groups"
            )
        if not self._is_none(bound.arguments["algorithm"]):
            raise NotImplementedError(
                "cuda.coop.numba_mlir.scan algorithm applies only to block groups"
            )
        if not self._is_none(bound.arguments["temp_storage"]):
            raise NotImplementedError(
                "cuda.coop.numba_mlir.scan temp_storage applies only to block groups"
            )
        if self._array_operand_state("scan", bound.arguments["value"]):
            raise NotImplementedError(
                "cuda.coop.numba_mlir.scan warp groups support one scalar value "
                "per lane"
            )

        scan_op = bound.arguments["scan_op"]
        default_sum = self._is_none(scan_op)
        if not default_sum:
            default_sum = ScanOp(self._constant(scan_op)).is_sum
        has_initial = not self._is_none(bound.arguments["initial_value"])
        has_valid_items = not self._is_none(bound.arguments["valid_items"])
        if has_valid_items:
            is_static, valid_items = self._try_constant(bound.arguments["valid_items"])
            if is_static:
                if isinstance(valid_items, bool) or not isinstance(
                    valid_items, Integral
                ):
                    raise TypeError(
                        "cuda.coop.numba_mlir.scan valid_items must be an "
                        "integer, not bool"
                    )
                group_size = group.static_size
                assert group_size is not None
                if not 1 <= int(valid_items) <= group_size:
                    raise ValueError(
                        "cuda.coop.numba_mlir.scan static valid_items must be "
                        f"between 1 and group size {group_size}"
                    )
        factory_operation = (
            f"{mode}_sum"
            if default_sum and not has_initial and not has_valid_items
            else f"{mode}_scan"
        )
        factory, factory_kwargs = self._scope_factory(group, factory_operation)
        if factory_operation.endswith("_scan"):
            factory_kwargs["scan_op"] = "+" if default_sum else scan_op
        if has_initial:
            factory_kwargs["initial_value"] = bound.arguments["initial_value"]
        if has_valid_items:
            factory_kwargs["valid_items"] = bound.arguments["valid_items"]
        if not self._is_none(bound.arguments["aggregate_output"]):
            factory_kwargs["warp_aggregate"] = bound.arguments["aggregate_output"]
        return self._rewritten_call(
            inst,
            factory=factory,
            args=[bound.arguments["value"]],
            kwargs=factory_kwargs,
            common_root_operation=(operation if is_common_root else None),
        )


__all__ = ["_ScanPlanning"]
