# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cross-family dtype and ThreadData factory inference.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _ThreadDataSpec,
    ir,
    normalize_dtype_param,
)


class _PayloadRewrite:
    def _infer_factory_kwargs_from_thread_data(
        self,
        op_name: str,
        runtime_args: list[ir.Var],
        allowed_factory_kwargs: set[str],
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        def factory_value(name: str):
            return factory_kwargs.get(name)

        def factory_kwarg_matches(name: str, actual, expected) -> bool:
            if name == "dtype":
                try:
                    actual = normalize_dtype_param(actual)
                    expected = normalize_dtype_param(expected)
                except (TypeError, ValueError):
                    pass
            return actual == expected

        def infer_kwarg(name: str, value) -> None:
            if name not in allowed_factory_kwargs or value is None:
                return
            if name in seen_factory_kwargs:
                if not factory_kwarg_matches(name, factory_kwargs[name], value):
                    raise CoopSinglePhaseRewriteError(
                        f"coop movement {op_name!r} {name} does not match the value inferred from ThreadData."
                    )
                return
            factory_kwargs[name] = value
            seen_factory_kwargs.add(name)

        def candidate(index: int) -> tuple[ir.Var | None, _ThreadDataSpec | None]:
            if not 0 <= index < len(runtime_args):
                return (None, None)
            value = runtime_args[index]
            if not isinstance(value, ir.Var):
                return (None, None)
            spec = self._resolve_thread_data_spec(value)
            if self._is_typed_group_payload_var(value) and (
                spec is None or spec.items_per_thread is None
            ):
                raise CoopSinglePhaseRewriteError(
                    f"coop movement {op_name!r} could not infer the static extent of a typed group payload"
                )
            return (value, spec)

        if op_name in {"load", "store", "warp_load", "warp_store"}:
            payload_var, payload_spec = candidate(1)
            if payload_spec is None:
                if op_name in {"store", "warp_store"}:
                    inferred_dtype = None
                    for arg in runtime_args[:2]:
                        if isinstance(arg, ir.Var):
                            inferred_dtype = self._resolve_var_dtype(arg)
                        if inferred_dtype is not None:
                            break
                    infer_kwarg("items_per_thread", 1)
                    infer_kwarg("dtype", inferred_dtype)
                return
            infer_kwarg("items_per_thread", payload_spec.items_per_thread)
            inferred_dtype = payload_spec.dtype
            if (
                inferred_dtype is None
                and runtime_args
                and isinstance(runtime_args[0], ir.Var)
            ):
                inferred_dtype = self._resolve_var_dtype(runtime_args[0])
            if inferred_dtype is None:
                inferred_dtype = factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and payload_var is not None:
                self._record_inferred_thread_data_dtype(payload_var, inferred_dtype)
            return
        if op_name in {"exchange", "warp_exchange"}:
            input_var, input_spec = candidate(0)
            second_var, second_spec = candidate(1)
            if input_spec is None and second_spec is None:
                return
            output_var = second_var
            output_spec = second_spec
            rank_spec = None
            valid_flag_spec = None
            uses_ranks = False
            uses_valid_flags = False
            out_of_place = True
            if op_name == "exchange":
                from .._lowering._exchange import (
                    BlockExchangeType,
                    _normalize_block_exchange_type,
                )

                exchange_type = _normalize_block_exchange_type(
                    factory_kwargs.get(
                        "block_exchange_type", BlockExchangeType.StripedToBlocked
                    )
                )
                uses_ranks = exchange_type in {
                    BlockExchangeType.ScatterToBlocked,
                    BlockExchangeType.ScatterToStriped,
                    BlockExchangeType.ScatterToStripedGuarded,
                    BlockExchangeType.ScatterToStripedFlagged,
                }
                uses_valid_flags = (
                    exchange_type == BlockExchangeType.ScatterToStripedFlagged
                )
                out_of_place = (
                    not uses_ranks
                    and len(runtime_args) == 2
                    or (
                        uses_ranks
                        and (not uses_valid_flags)
                        and (len(runtime_args) == 3)
                    )
                    or (uses_valid_flags and len(runtime_args) == 4)
                )
                if uses_ranks:
                    _, rank_spec = candidate(2 if out_of_place else 1)
                if uses_valid_flags:
                    _, valid_flag_spec = candidate(3 if out_of_place else 2)
            else:
                from .._lowering._exchange import (
                    WarpExchangeType,
                    _normalize_warp_exchange_type,
                )

                exchange_type = _normalize_warp_exchange_type(
                    factory_kwargs.get(
                        "warp_exchange_type", WarpExchangeType.StripedToBlocked
                    )
                )
                uses_ranks = exchange_type == WarpExchangeType.ScatterToStriped
                out_of_place = not uses_ranks or len(runtime_args) == 3
                if uses_ranks:
                    _, rank_spec = candidate(2 if out_of_place else 1)
            if not out_of_place:
                output_var = None
                output_spec = None
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "output", output_spec
            )
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "ranks", rank_spec
            )
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "valid_flags", valid_flag_spec
            )
            if (
                input_spec is not None
                and output_spec is not None
                and (input_spec.dtype is not None)
                and (output_spec.dtype is not None)
                and (input_spec.dtype != output_spec.dtype)
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop exchange requires input/output arrays to have matching dtype."
                )
            inferred_dtype = input_spec.dtype if input_spec is not None else None
            if inferred_dtype is None and output_spec is not None:
                inferred_dtype = output_spec.dtype
            if inferred_dtype is None:
                inferred_dtype = factory_value("dtype")
            extent = input_spec.items_per_thread if input_spec is not None else None
            if extent is None and output_spec is not None:
                extent = output_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None:
                if input_var is not None:
                    self._record_inferred_thread_data_dtype(input_var, inferred_dtype)
                if output_var is not None:
                    self._record_inferred_thread_data_dtype(output_var, inferred_dtype)
            return
        if op_name == "shuffle":
            if len(runtime_args) == 1:
                value = runtime_args[0]
                inferred_dtype = (
                    self._resolve_var_dtype(value)
                    if isinstance(value, ir.Var)
                    else None
                )
                infer_kwarg("dtype", inferred_dtype or factory_value("dtype"))
                return
            input_var, input_spec = candidate(0)
            output_var, output_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "output", output_spec
            )
            extent = input_spec.items_per_thread if input_spec is not None else None
            if extent is None and output_spec is not None:
                extent = output_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            inferred_dtype = input_spec.dtype if input_spec is not None else None
            if inferred_dtype is None and output_spec is not None:
                inferred_dtype = output_spec.dtype
            if inferred_dtype is None and input_var is not None:
                inferred_dtype = self._resolve_var_dtype(input_var)
            infer_kwarg("dtype", inferred_dtype or factory_value("dtype"))
            if inferred_dtype is not None:
                if input_var is not None:
                    self._record_inferred_thread_data_dtype(input_var, inferred_dtype)
                if output_var is not None:
                    self._record_inferred_thread_data_dtype(output_var, inferred_dtype)


__all__ = ["_PayloadRewrite"]
