# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exchange IR planning.

This mixin owns only its primitive-family IR rewrite. Shared provenance,
launch facts, caches, and final orchestration remain in the group planner.
"""

from ._group_planner_support import (
    _PAYLOAD_DTYPE_LIKE,
    Any,
    ThreadGroup,
    inspect,
    ir,
)


class _ExchangePlanning:
    def _lower_exchange(
        self,
        inst: ir.Assign,
        *,
        group: ThreadGroup,
        bound: inspect.BoundArguments,
        is_common_root: bool,
    ) -> list[Any]:
        self._reject_extra_root_arguments("exchange", bound)
        if group.kind not in {"block", "warp", "threads_within_warp"}:
            raise NotImplementedError(
                "cuda.coop.numba_mlir.exchange currently lowers only block, physical-warp, and logical-warp groups"
            )
        mode = self._constant(bound.arguments["mode"])
        if hasattr(mode, "value"):
            mode = mode.value
        if not isinstance(mode, str):
            raise TypeError(
                "cuda.coop.numba_mlir.exchange mode must be a compile-time string"
            )
        time_slicing = self._constant(bound.arguments["warp_time_slicing"])
        if not isinstance(time_slicing, bool):
            raise TypeError(
                "cuda.coop.numba_mlir.exchange warp_time_slicing must be a compile-time bool"
            )
        if group.kind == "block":
            from .._lowering._exchange import BlockExchangeType

            exchange_types = {
                "striped_to_blocked": BlockExchangeType.StripedToBlocked,
                "blocked_to_striped": BlockExchangeType.BlockedToStriped,
                "warp_striped_to_blocked": BlockExchangeType.WarpStripedToBlocked,
                "blocked_to_warp_striped": BlockExchangeType.BlockedToWarpStriped,
                "scatter_to_blocked": BlockExchangeType.ScatterToBlocked,
                "scatter_to_striped": BlockExchangeType.ScatterToStriped,
                "scatter_to_striped_guarded": BlockExchangeType.ScatterToStripedGuarded,
                "scatter_to_striped_flagged": BlockExchangeType.ScatterToStripedFlagged,
            }
            exchange_type_name = "block_exchange_type"
        else:
            from .._lowering._exchange import WarpExchangeType

            exchange_types = {
                "striped_to_blocked": WarpExchangeType.StripedToBlocked,
                "blocked_to_striped": WarpExchangeType.BlockedToStriped,
                "scatter_to_striped": WarpExchangeType.ScatterToStriped,
            }
            exchange_type_name = "warp_exchange_type"
            if time_slicing:
                raise ValueError(
                    "cuda.coop.numba_mlir.exchange warp_time_slicing applies only to block groups"
                )
        try:
            exchange_type = exchange_types[mode]
        except KeyError as exc:
            choices = ", ".join(exchange_types)
            raise ValueError(
                f"cuda.coop.numba_mlir.exchange mode must be one of: {choices}"
            ) from exc
        uses_ranks = mode.startswith("scatter_to_")
        uses_valid_flags = mode == "scatter_to_striped_flagged"
        has_ranks = not self._is_none(bound.arguments["ranks"])
        has_valid_flags = not self._is_none(bound.arguments["valid_flags"])
        if uses_ranks != has_ranks:
            requirement = "requires" if uses_ranks else "does not accept"
            raise ValueError(
                f"cuda.coop.numba_mlir.exchange {mode} {requirement} ranks"
            )
        if uses_valid_flags != has_valid_flags:
            requirement = "requires" if uses_valid_flags else "does not accept"
            raise ValueError(
                f"cuda.coop.numba_mlir.exchange {mode} {requirement} valid_flags"
            )
        factory, factory_kwargs = self._scope_factory(group, "exchange")
        factory_kwargs[exchange_type_name] = exchange_type
        if time_slicing:
            factory_kwargs["warp_time_slicing"] = True
        if is_common_root:
            factory_kwargs["_common_root_operation"] = "exchange"
        statements: list[Any] = []
        scope = inst.target.scope
        loc = inst.loc
        value = self._value_var(
            statements,
            scope=scope,
            loc=loc,
            stem="exchange_value",
            value=bound.arguments["value"],
        )
        if not self._array_operand_state("exchange", value):
            raise TypeError(
                "cuda.coop.numba_mlir.exchange requires a fixed-size ThreadData or local-array payload"
            )
        if is_common_root and (
            not self._thread_data_operand_state("exchange", "value", value)
        ):
            raise TypeError(
                "cuda.coop.exchange requires a fixed-size ThreadData payload in the portable API; use cuda.coop.numba_mlir for backend-qualified local-array payload support"
            )
        result_payload = self._typed_payload_like(
            statements,
            scope=scope,
            loc=loc,
            stem="exchange_result",
            prototype=value,
            is_array=True,
            dtype_policy=_PAYLOAD_DTYPE_LIKE,
        )
        runtime_args = [value, result_payload]
        if uses_ranks:
            ranks = bound.arguments["ranks"]
            if not self._array_operand_state("exchange", ranks):
                raise TypeError(
                    "cuda.coop.numba_mlir.exchange ranks must be a fixed-size ThreadData or local-array payload"
                )
            runtime_args.append(ranks)
        if uses_valid_flags:
            valid_flags = bound.arguments["valid_flags"]
            if not self._array_operand_state("exchange", valid_flags):
                raise TypeError(
                    "cuda.coop.numba_mlir.exchange valid_flags must be a fixed-size ThreadData or local-array payload"
                )
            runtime_args.append(valid_flags)
        call_statements = self._rewritten_call(
            inst,
            factory=factory,
            args=runtime_args,
            kwargs=factory_kwargs,
            return_alias=result_payload,
        )
        call_statements.pop()
        statements.extend(call_statements)
        statements.append(ir.Assign(result_payload, inst.target, loc))
        return statements


__all__ = ["_ExchangePlanning"]
