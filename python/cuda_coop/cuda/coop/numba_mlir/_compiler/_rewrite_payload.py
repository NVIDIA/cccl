# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cross-family dtype and ThreadData factory inference.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _dtype_values_match,
    _ThreadDataSpec,
    ir,
    normalize_dtype_param,
    np,
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
        common_operation = {
            "_common_radix_rank": "radix_rank",
            "_common_radix_sort_keys": "radix_sort_keys",
            "_common_radix_sort_pairs": "radix_sort_pairs",
        }.get(op_name)
        op_name = {
            "_common_radix_rank": "radix_rank",
            "_common_radix_sort_keys": "radix_sort_keys",
            "_common_radix_sort_pairs": "radix_sort_pairs",
        }.get(op_name, op_name)

        common_topk_operation = {
            "_common_topk_max_keys": "topk_max_keys",
            "_common_topk_max_pairs": "topk_max_pairs",
            "_common_topk_min_keys": "topk_min_keys",
            "_common_topk_min_pairs": "topk_min_pairs",
        }.get(op_name)
        op_name = {
            "_common_topk_max_keys": "topk_max_keys",
            "_common_topk_max_pairs": "topk_max_pairs",
            "_common_topk_min_keys": "topk_min_keys",
            "_common_topk_min_pairs": "topk_min_pairs",
            "_qualified_group_topk_max_keys": "topk_max_keys",
            "_qualified_group_topk_max_pairs": "topk_max_pairs",
            "_qualified_group_topk_min_keys": "topk_min_keys",
            "_qualified_group_topk_min_pairs": "topk_min_pairs",
        }.get(op_name, op_name)

        def factory_value(name: str):
            return factory_kwargs.get(name)

        def validate_integer_key_dtype(dtype):
            if dtype is None:
                return None
            from ._parameters import _validate_common_integer_key_dtype

            try:
                return _validate_common_integer_key_dtype(
                    dtype, operation=common_operation or op_name
                )
            except (TypeError, ValueError) as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc

        def validate_numeric_value_dtype(dtype):
            if dtype is None:
                return None
            from ._parameters import _validate_common_numeric_dtype

            try:
                return _validate_common_numeric_dtype(
                    dtype,
                    operation=common_operation or op_name,
                    parameter="value",
                )
            except (TypeError, ValueError) as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc

        def factory_kwarg_matches(name: str, actual, expected) -> bool:
            if name in {
                "dtype",
                "keys",
                "values",
            }:
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

        def validate_common_key_dtype(dtype):
            if common_topk_operation is None or dtype is None:
                return dtype
            from ._parameters import _validate_common_integer_key_dtype

            try:
                return _validate_common_integer_key_dtype(
                    dtype,
                    operation=common_topk_operation,
                )
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc

        def validate_common_value_dtype(dtype):
            if common_topk_operation is None or dtype is None:
                return dtype
            from ._parameters import _validate_common_numeric_dtype

            try:
                return _validate_common_numeric_dtype(
                    dtype,
                    operation=common_topk_operation,
                    parameter="value",
                )
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc

        if op_name in {"topk_max_keys", "topk_min_keys"}:
            key_var, key_spec = candidate(0)
            if key_spec is not None:
                infer_kwarg("items_per_thread", key_spec.items_per_thread)
            key_dtype = key_spec.dtype if key_spec is not None else None
            if key_dtype is None and key_var is not None:
                key_dtype = self._resolve_var_dtype(key_var)
            if key_dtype is None:
                key_dtype = factory_value("dtype")
            key_dtype = validate_common_key_dtype(key_dtype)
            infer_kwarg("dtype", key_dtype)
            if key_dtype is not None and key_var is not None:
                self._record_inferred_thread_data_dtype(key_var, key_dtype)
            return

        if op_name in {"topk_max_pairs", "topk_min_pairs"}:
            key_var, key_spec = candidate(0)
            value_var, value_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name,
                "key",
                key_spec,
                "value",
                value_spec,
            )
            extent = key_spec.items_per_thread if key_spec is not None else None
            if extent is None and value_spec is not None:
                extent = value_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            key_dtype = key_spec.dtype if key_spec is not None else None
            value_dtype = value_spec.dtype if value_spec is not None else None
            if key_dtype is None and key_var is not None:
                key_dtype = self._resolve_var_dtype(key_var)
            if value_dtype is None and value_var is not None:
                value_dtype = self._resolve_var_dtype(value_var)
            if key_dtype is None:
                key_dtype = factory_value("keys")
            if value_dtype is None:
                value_dtype = factory_value("values")
            key_dtype = validate_common_key_dtype(key_dtype)
            value_dtype = validate_common_value_dtype(value_dtype)
            infer_kwarg("keys", key_dtype)
            infer_kwarg("values", value_dtype)
            if key_dtype is not None and key_var is not None:
                self._record_inferred_thread_data_dtype(key_var, key_dtype)
            if value_dtype is not None and value_var is not None:
                self._record_inferred_thread_data_dtype(value_var, value_dtype)
            return

        if op_name in {
            "group_reduce",
            "block_reduce_builtin",
            "reduce",
            "sum",
            "warp_reduce_builtin",
            "warp_reduce",
            "warp_sum",
        }:
            payload_var, payload_spec = candidate(0)
            if payload_spec is not None:
                infer_kwarg("items_per_thread", payload_spec.items_per_thread)
            inferred_dtype = payload_spec.dtype if payload_spec is not None else None
            if inferred_dtype is None and payload_var is not None:
                inferred_dtype = self._resolve_var_dtype(payload_var)
            if inferred_dtype is None:
                inferred_dtype = factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)
            if inferred_dtype is not None and payload_var is not None:
                self._record_inferred_thread_data_dtype(payload_var, inferred_dtype)
            return

        if op_name == "scan":
            input_var, input_spec = candidate(0)
            output_var, output_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "input", input_spec, "output", output_spec
            )
            input_dtype = input_spec.dtype if input_spec is not None else None
            output_dtype = output_spec.dtype if output_spec is not None else None
            if input_dtype is None and input_var is not None:
                input_dtype = self._resolve_var_dtype(input_var)
            if output_dtype is None and output_var is not None:
                output_dtype = self._resolve_var_dtype(output_var)
            if (
                input_dtype is not None
                and output_dtype is not None
                and not _dtype_values_match(input_dtype, output_dtype)
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop scan requires input/output arrays to have matching dtype."
                )
            inferred_dtype = input_dtype
            if inferred_dtype is None:
                inferred_dtype = output_dtype
            if inferred_dtype is None:
                inferred_dtype = factory_value("dtype")
            extent = input_spec.items_per_thread if input_spec is not None else None
            if extent is None and output_spec is not None:
                extent = output_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            infer_kwarg("dtype", inferred_dtype)
            for payload_var in (input_var, output_var):
                if inferred_dtype is not None and payload_var is not None:
                    self._record_inferred_thread_data_dtype(payload_var, inferred_dtype)
            if factory_kwargs.get("block_aggregate"):
                aggregate_var, aggregate_spec = candidate(2)
                if aggregate_spec is None or aggregate_spec.items_per_thread != 1:
                    raise CoopSinglePhaseRewriteError(
                        "coop scan block_aggregate must be a one-item "
                        "ThreadData or local array."
                    )
                aggregate_dtype = aggregate_spec.dtype
                if aggregate_dtype is None and aggregate_var is not None:
                    aggregate_dtype = self._resolve_var_dtype(aggregate_var)
                if (
                    inferred_dtype is not None
                    and aggregate_dtype is not None
                    and not _dtype_values_match(inferred_dtype, aggregate_dtype)
                ):
                    raise CoopSinglePhaseRewriteError(
                        "coop scan block_aggregate dtype must match the input dtype."
                    )
                if aggregate_var is not None and inferred_dtype is not None:
                    self._record_inferred_thread_data_dtype(
                        aggregate_var, inferred_dtype
                    )
            return

        if op_name in {
            "warp_exclusive_sum",
            "warp_inclusive_sum",
            "warp_exclusive_scan",
            "warp_inclusive_scan",
        }:
            value_var, value_spec = candidate(0)
            inferred_dtype = value_spec.dtype if value_spec is not None else None
            if inferred_dtype is None and value_var is not None:
                inferred_dtype = self._resolve_var_dtype(value_var)
            if inferred_dtype is None:
                inferred_dtype = factory_value("dtype")
            infer_kwarg("dtype", inferred_dtype)
            aggregate_index = None
            if factory_kwargs.get("warp_aggregate"):
                aggregate_index = (
                    2
                    if factory_kwargs.get("valid_items")
                    and op_name in {"warp_exclusive_scan", "warp_inclusive_scan"}
                    else 1
                )
            if aggregate_index is not None:
                aggregate_var, aggregate_spec = candidate(aggregate_index)
                if aggregate_spec is None or aggregate_spec.items_per_thread != 1:
                    raise CoopSinglePhaseRewriteError(
                        "coop scan warp_aggregate must be a one-item "
                        "ThreadData or local array."
                    )
                aggregate_dtype = aggregate_spec.dtype
                if aggregate_dtype is None and aggregate_var is not None:
                    aggregate_dtype = self._resolve_var_dtype(aggregate_var)
                if (
                    inferred_dtype is not None
                    and aggregate_dtype is not None
                    and not _dtype_values_match(inferred_dtype, aggregate_dtype)
                ):
                    raise CoopSinglePhaseRewriteError(
                        "coop scan warp_aggregate dtype must match the input dtype."
                    )
                if aggregate_var is not None and inferred_dtype is not None:
                    self._record_inferred_thread_data_dtype(
                        aggregate_var, inferred_dtype
                    )
            return

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
        if op_name in {"radix_sort_keys", "radix_sort_keys_descending"}:
            keys_var, keys_spec = candidate(0)
            if keys_spec is None:
                return
            infer_kwarg("items_per_thread", keys_spec.items_per_thread)
            key_dtype = keys_spec.dtype
            if key_dtype is None and keys_var is not None:
                key_dtype = self._resolve_var_dtype(keys_var)
            if key_dtype is None:
                key_dtype = factory_value("dtype")
            key_dtype = validate_integer_key_dtype(key_dtype)
            infer_kwarg("dtype", key_dtype)
            if key_dtype is not None and keys_var is not None:
                self._record_inferred_thread_data_dtype(keys_var, key_dtype)
            return
        if op_name in {"radix_sort_pairs", "radix_sort_pairs_descending"}:
            keys_var, keys_spec = candidate(0)
            values_var, values_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "keys", keys_spec, "values", values_spec
            )
            extent = keys_spec.items_per_thread if keys_spec is not None else None
            if extent is None and values_spec is not None:
                extent = values_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            key_dtype = keys_spec.dtype if keys_spec is not None else None
            value_dtype = values_spec.dtype if values_spec is not None else None
            if key_dtype is None and keys_var is not None:
                key_dtype = self._resolve_var_dtype(keys_var)
            if value_dtype is None and values_var is not None:
                value_dtype = self._resolve_var_dtype(values_var)
            if key_dtype is None:
                key_dtype = factory_value("key_dtype")
            if value_dtype is None:
                value_dtype = factory_value("value_dtype")
            key_dtype = validate_integer_key_dtype(key_dtype)
            value_dtype = validate_numeric_value_dtype(value_dtype)
            infer_kwarg("key_dtype", key_dtype)
            infer_kwarg("value_dtype", value_dtype)
            if key_dtype is not None and keys_var is not None:
                self._record_inferred_thread_data_dtype(keys_var, key_dtype)
            if value_dtype is not None and values_var is not None:
                self._record_inferred_thread_data_dtype(values_var, value_dtype)
            return
        if op_name == "radix_rank":
            from numba_cuda_mlir import types as numba_mlir_types

            def is_int32_dtype(dtype) -> bool:
                if dtype == numba_mlir_types.int32:
                    return True
                try:
                    return np.dtype(dtype) == np.dtype(np.int32)
                except (TypeError, ValueError):
                    return False

            keys_var, keys_spec = candidate(0)
            ranks_var, ranks_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name, "keys", keys_spec, "ranks", ranks_spec
            )
            extent = keys_spec.items_per_thread if keys_spec is not None else None
            if extent is None and ranks_spec is not None:
                extent = ranks_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            key_dtype = keys_spec.dtype if keys_spec is not None else None
            if key_dtype is None and keys_var is not None:
                key_dtype = self._resolve_var_dtype(keys_var)
            if key_dtype is None:
                key_dtype = factory_value("dtype")
            key_dtype = validate_integer_key_dtype(key_dtype)
            infer_kwarg("dtype", key_dtype)
            if key_dtype is not None and keys_var is not None:
                self._record_inferred_thread_data_dtype(keys_var, key_dtype)
            if ranks_spec is not None and ranks_var is not None:
                if ranks_spec.dtype is not None and not is_int32_dtype(
                    ranks_spec.dtype
                ):
                    raise CoopSinglePhaseRewriteError(
                        "coop single-phase 'radix_rank' requires ranks dtype int32."
                    )
                self._record_inferred_thread_data_dtype(
                    ranks_var, numba_mlir_types.int32
                )
            if factory_kwargs.get("exclusive_digit_prefix"):
                prefix_var, prefix_spec = candidate(2)
                if prefix_spec is None or prefix_var is None:
                    raise CoopSinglePhaseRewriteError(
                        "radix_rank exclusive_digit_prefix must be a local array"
                    )
                if prefix_spec.dtype is not None and not is_int32_dtype(
                    prefix_spec.dtype
                ):
                    raise CoopSinglePhaseRewriteError(
                        "radix_rank exclusive_digit_prefix dtype must be int32"
                    )
                self._record_inferred_thread_data_dtype(
                    prefix_var, numba_mlir_types.int32
                )
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
        if op_name in {"merge_sort_keys", "warp_merge_sort_keys"}:
            keys_var, keys_spec = candidate(0)
            if keys_spec is None:
                return
            infer_kwarg("items_per_thread", keys_spec.items_per_thread)
            key_dtype = keys_spec.dtype
            if key_dtype is None and keys_var is not None:
                key_dtype = self._resolve_var_dtype(keys_var)
            if key_dtype is None and keys_var is not None:
                key_dtype = self._infer_thread_data_dtype_from_provenance_writes(
                    keys_var
                )
            if key_dtype is None:
                key_dtype = factory_value("dtype")
            infer_kwarg("dtype", key_dtype)
            if key_dtype is not None and keys_var is not None:
                self._record_inferred_thread_data_dtype(keys_var, key_dtype)
            return
        if op_name in {"merge_sort_pairs", "warp_merge_sort_pairs"}:
            keys_var, keys_spec = candidate(0)
            values_var, values_spec = candidate(1)
            self._require_matching_items_per_thread(
                op_name,
                "keys",
                keys_spec,
                "values",
                values_spec,
            )
            extent = keys_spec.items_per_thread if keys_spec is not None else None
            if extent is None and values_spec is not None:
                extent = values_spec.items_per_thread
            infer_kwarg("items_per_thread", extent)
            key_dtype = keys_spec.dtype if keys_spec is not None else None
            value_dtype = values_spec.dtype if values_spec is not None else None
            if key_dtype is None and keys_var is not None:
                key_dtype = self._resolve_var_dtype(keys_var)
            if value_dtype is None and values_var is not None:
                value_dtype = self._resolve_var_dtype(values_var)
            if key_dtype is None and keys_var is not None:
                key_dtype = self._infer_thread_data_dtype_from_provenance_writes(
                    keys_var
                )
            if value_dtype is None and values_var is not None:
                value_dtype = self._infer_thread_data_dtype_from_provenance_writes(
                    values_var
                )
            if key_dtype is None:
                key_dtype = factory_value("keys")
            if value_dtype is None:
                value_dtype = factory_value("values")
            infer_kwarg("keys", key_dtype)
            infer_kwarg("values", value_dtype)
            if key_dtype is not None and keys_var is not None:
                self._record_inferred_thread_data_dtype(keys_var, key_dtype)
            if value_dtype is not None and values_var is not None:
                self._record_inferred_thread_data_dtype(values_var, value_dtype)
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
