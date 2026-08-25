# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Radix control validation and rewrite finalization.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_payload import PayloadInference
from ._rewrite_support import (
    _UNRESOLVED,
    CoopSinglePhaseRewriteError,
    block_radix_rank_bins_per_thread,
    ir,
    normalize_dim_param,
    normalize_radix_order,
    np,
    operator,
    resolve_static_radix_end_bit,
)


class _RadixRewrite:
    def _infer_radix_payload(self, inference: PayloadInference) -> None:
        """Infer radix key/value payload metadata and rank output types."""

        def validate_integer_key_dtype(dtype):
            if dtype is None:
                return None
            from ._parameters import _validate_common_integer_key_dtype

            try:
                return _validate_common_integer_key_dtype(
                    dtype,
                    operation=(inference.portable_op_name or inference.op_name),
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
                    operation=(inference.portable_op_name or inference.op_name),
                    parameter="value",
                )
            except (TypeError, ValueError) as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc

        if inference.op_name in {
            "radix_sort_keys",
            "radix_sort_keys_descending",
        }:
            keys_var, keys_spec = inference.candidate(0)
            if keys_spec is None:
                return
            inference.infer_kwarg("items_per_thread", keys_spec.items_per_thread)
            key_dtype = keys_spec.dtype
            if key_dtype is None and keys_var is not None:
                key_dtype = self._resolve_var_dtype(keys_var)
            if key_dtype is None:
                key_dtype = inference.factory_value("dtype")
            key_dtype = validate_integer_key_dtype(key_dtype)
            inference.infer_kwarg("dtype", key_dtype)
            if key_dtype is not None and keys_var is not None:
                self._record_inferred_thread_data_dtype(keys_var, key_dtype)
            return

        if inference.op_name in {
            "radix_sort_pairs",
            "radix_sort_pairs_descending",
        }:
            keys_var, keys_spec = inference.candidate(0)
            values_var, values_spec = inference.candidate(1)
            self._require_matching_items_per_thread(
                inference.op_name,
                "keys",
                keys_spec,
                "values",
                values_spec,
            )
            extent = keys_spec.items_per_thread if keys_spec is not None else None
            if extent is None and values_spec is not None:
                extent = values_spec.items_per_thread
            inference.infer_kwarg("items_per_thread", extent)
            key_dtype = keys_spec.dtype if keys_spec is not None else None
            value_dtype = values_spec.dtype if values_spec is not None else None
            if key_dtype is None and keys_var is not None:
                key_dtype = self._resolve_var_dtype(keys_var)
            if value_dtype is None and values_var is not None:
                value_dtype = self._resolve_var_dtype(values_var)
            if key_dtype is None:
                key_dtype = inference.factory_value("key_dtype")
            if value_dtype is None:
                value_dtype = inference.factory_value("value_dtype")
            key_dtype = validate_integer_key_dtype(key_dtype)
            value_dtype = validate_numeric_value_dtype(value_dtype)
            inference.infer_kwarg("key_dtype", key_dtype)
            inference.infer_kwarg("value_dtype", value_dtype)
            if key_dtype is not None and keys_var is not None:
                self._record_inferred_thread_data_dtype(keys_var, key_dtype)
            if value_dtype is not None and values_var is not None:
                self._record_inferred_thread_data_dtype(values_var, value_dtype)
            return

        from numba_cuda_mlir import types as numba_mlir_types

        def is_int32_dtype(dtype) -> bool:
            if dtype == numba_mlir_types.int32:
                return True
            try:
                return np.dtype(dtype) == np.dtype(np.int32)
            except (TypeError, ValueError):
                return False

        keys_var, keys_spec = inference.candidate(0)
        ranks_var, ranks_spec = inference.candidate(1)
        self._require_matching_items_per_thread(
            inference.op_name, "keys", keys_spec, "ranks", ranks_spec
        )
        extent = keys_spec.items_per_thread if keys_spec is not None else None
        if extent is None and ranks_spec is not None:
            extent = ranks_spec.items_per_thread
        inference.infer_kwarg("items_per_thread", extent)
        key_dtype = keys_spec.dtype if keys_spec is not None else None
        if key_dtype is None and keys_var is not None:
            key_dtype = self._resolve_var_dtype(keys_var)
        if key_dtype is None:
            key_dtype = inference.factory_value("dtype")
        key_dtype = validate_integer_key_dtype(key_dtype)
        inference.infer_kwarg("dtype", key_dtype)
        if key_dtype is not None and keys_var is not None:
            self._record_inferred_thread_data_dtype(keys_var, key_dtype)
        if ranks_spec is not None and ranks_var is not None:
            if ranks_spec.dtype is not None and not is_int32_dtype(ranks_spec.dtype):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'radix_rank' requires ranks dtype int32."
                )
            self._record_inferred_thread_data_dtype(ranks_var, numba_mlir_types.int32)
        if inference.factory_kwargs.get("exclusive_digit_prefix"):
            prefix_var, prefix_spec = inference.candidate(2)
            if prefix_spec is None or prefix_var is None:
                raise CoopSinglePhaseRewriteError(
                    "radix_rank exclusive_digit_prefix must be a local array"
                )
            if prefix_spec.dtype is not None and not is_int32_dtype(prefix_spec.dtype):
                raise CoopSinglePhaseRewriteError(
                    "radix_rank exclusive_digit_prefix dtype must be int32"
                )
            self._record_inferred_thread_data_dtype(prefix_var, numba_mlir_types.int32)

    def _finalize_radix_rank_factory_kwargs(
        self,
        *,
        op_name: str,
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        if runtime_arg_count not in {2, 3}:
            raise CoopSinglePhaseRewriteError(
                "coop single-phase 'radix_rank' runtime argument count must "
                "be one of {2, 3}."
            )
        scope_name = (
            "cuda.coop" if op_name == "_common_radix_rank" else "cuda.coop.numba_mlir"
        )

        def static_index(name: str, value: object) -> int:
            if isinstance(value, (bool, np.bool_)):
                raise CoopSinglePhaseRewriteError(
                    f"{scope_name}.radix_rank {name} must be an integer"
                )
            try:
                return operator.index(value)
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(
                    f"{scope_name}.radix_rank {name} must be an integer"
                ) from exc

        begin_bit = static_index("begin_bit", factory_kwargs.get("begin_bit", 0))
        dtype = factory_kwargs.get("dtype")
        bitwidth = getattr(dtype, "bitwidth", None)
        if bitwidth is not None:
            bitwidth = int(bitwidth)
        explicit_end_bit = factory_kwargs.get("end_bit")
        if explicit_end_bit is not None:
            explicit_end_bit = static_index("end_bit", explicit_end_bit)
        try:
            end_bit = resolve_static_radix_end_bit(
                begin_bit=begin_bit,
                end_bit=explicit_end_bit,
                bit_width=bitwidth,
                default_radix_bits=4,
                clamp_default=False,
            )
        except ValueError as exc:
            raise CoopSinglePhaseRewriteError(f"{scope_name}.radix_rank {exc}") from exc
        if end_bit - begin_bit > 8:
            raise CoopSinglePhaseRewriteError(
                f"{scope_name}.radix_rank bit width must be <= 8"
            )
        factory_kwargs["begin_bit"] = begin_bit
        factory_kwargs["end_bit"] = end_bit
        seen_factory_kwargs.update({"begin_bit", "end_bit"})
        if "descending" in seen_factory_kwargs:
            try:
                factory_kwargs["descending"] = normalize_radix_order(
                    factory_kwargs["descending"]
                ).descending
            except ValueError as exc:
                raise CoopSinglePhaseRewriteError(
                    f"{scope_name}.radix_rank descending must be a bool"
                ) from exc

    def _validate_radix_rank_exclusive_digit_prefix_extent(
        self,
        *,
        op_name: str,
        control_vars: dict[str, ir.Var],
        factory_kwargs: dict[str, object],
    ) -> None:
        if not factory_kwargs.get("exclusive_digit_prefix"):
            return
        scope_name = (
            "cuda.coop" if op_name == "_common_radix_rank" else "cuda.coop.numba_mlir"
        )
        prefix_var = control_vars.get("exclusive_digit_prefix")
        if not isinstance(prefix_var, ir.Var):
            raise CoopSinglePhaseRewriteError(
                f"{scope_name}.radix_rank internal rewrite error: "
                "exclusive_digit_prefix has no runtime argument"
            )
        prefix_spec = self._resolve_thread_data_spec(prefix_var)
        threads = factory_kwargs.get("threads_per_block")
        if (
            prefix_spec is None
            or prefix_spec.items_per_thread is None
            or threads is None
        ):
            return
        block_dim = normalize_dim_param(threads)
        block_threads = block_dim.x * block_dim.y * block_dim.z
        begin_bit = int(factory_kwargs["begin_bit"])
        end_bit = int(factory_kwargs["end_bit"])
        expected = block_radix_rank_bins_per_thread(
            end_bit - begin_bit,
            block_threads,
        )
        if prefix_spec.items_per_thread != expected:
            raise CoopSinglePhaseRewriteError(
                f"{scope_name}.radix_rank exclusive_digit_prefix must contain "
                f"{expected} items per thread"
            )

    def _radix_sort_runtime_constant_replacements(
        self,
        *,
        op_name: str,
        runtime_args: list[ir.Var],
        runtime_only_kw_vars: dict[str, ir.Var],
        factory_kwargs: dict[str, object],
    ) -> tuple[tuple[int, object], ...]:
        radix_sort_operations = {
            "_common_radix_sort_keys",
            "_common_radix_sort_pairs",
            "radix_sort_keys",
            "radix_sort_keys_descending",
            "radix_sort_pairs",
            "radix_sort_pairs_descending",
        }
        if op_name not in radix_sort_operations:
            return ()
        begin_var = runtime_only_kw_vars.get("begin_bit")
        end_var = runtime_only_kw_vars.get("end_bit")
        if begin_var is None and end_var is None:
            return ()
        if begin_var is None or end_var is None:
            return ()

        common_root = op_name.startswith("_common_")
        public_operation = (
            "radix_sort_pairs" if "pairs" in op_name else "radix_sort_keys"
        )
        scope_name = "cuda.coop" if common_root else "cuda.coop.numba_mlir"
        prefix = f"{scope_name}.{public_operation}"

        def static_bound(name: str, value_ref: ir.Var) -> int | None:
            value = self._resolve_factory_kwarg_value(name, value_ref)
            if value is _UNRESOLVED:
                from numba_cuda_mlir import types as numba_mlir_types

                value_type = self._resolve_var_numba_type(value_ref)
                if value_type is None:
                    value_type = self._resolve_var_dtype(value_ref)
                if isinstance(value_type, numba_mlir_types.Boolean) or not isinstance(
                    value_type, numba_mlir_types.Integer
                ):
                    raise CoopSinglePhaseRewriteError(
                        f"{prefix} {name} must have an integer dtype"
                    )
                return None
            if value is None:
                return None
            if isinstance(value, (bool, np.bool_)):
                raise CoopSinglePhaseRewriteError(f"{prefix} {name} must be an integer")
            try:
                return operator.index(value)
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} {name} must be an integer"
                ) from exc

        static_begin = static_bound("begin_bit", begin_var)
        dtype = factory_kwargs.get("dtype", factory_kwargs.get("key_dtype"))
        bit_width = getattr(dtype, "bitwidth", None)
        if bit_width is not None:
            bit_width = int(bit_width)
        static_end = static_bound("end_bit", end_var)
        replacements: tuple[tuple[int, object], ...] = ()
        end_value = self._resolve_factory_kwarg_value("end_bit", end_var)
        if end_value is None:
            if bit_width is None:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} end_bit must be provided when the key dtype "
                    "bit width cannot be inferred"
                )
            static_end = bit_width
            end_index = next(
                index
                for index, argument in enumerate(runtime_args)
                if argument is end_var or argument.name == end_var.name
            )
            replacements = ((end_index, static_end),)
        if static_begin is not None:
            if static_begin < 0:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} begin_bit must be non-negative"
                )
            if bit_width is not None and static_begin >= bit_width:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} begin_bit must be < {bit_width}"
                )
        if static_end is not None:
            if static_end < 1:
                raise CoopSinglePhaseRewriteError(f"{prefix} end_bit must be positive")
            if bit_width is not None and static_end > bit_width:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} end_bit must be <= {bit_width}"
                )
        if (
            static_begin is not None
            and static_end is not None
            and static_end <= static_begin
        ):
            raise CoopSinglePhaseRewriteError(
                f"{prefix} end_bit must be greater than begin_bit"
            )
        return replacements

    def _validate_integer_runtime_controls(
        self,
        *,
        op_name: str,
        runtime_args: list[ir.Var],
        factory_kwargs: dict[str, object],
    ) -> None:
        """Reject bool and noninteger partial-tile controls before codegen."""

        parameter = None
        index = None
        if op_name in {"reduce", "sum", "block_reduce_builtin"} and factory_kwargs.get(
            "num_valid"
        ):
            parameter, index = "valid_items", 1
        elif op_name in {
            "warp_reduce",
            "warp_sum",
            "warp_reduce_builtin",
        } and factory_kwargs.get("valid_items"):
            parameter, index = "valid_items", 1
        elif op_name in {"warp_exclusive_scan", "warp_inclusive_scan"} and (
            factory_kwargs.get("valid_items")
        ):
            parameter, index = "valid_items", 1
        elif op_name == "adjacent_difference" and factory_kwargs.get("valid_items"):
            parameter, index = "valid_items", 2
        if parameter is None or index is None or index >= len(runtime_args):
            return
        value = runtime_args[index]
        if not isinstance(value, ir.Var):
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase '{op_name}' {parameter} must be an integer"
            )
        from numba_cuda_mlir import types as numba_mlir_types

        value_type = self._resolve_var_numba_type(value)
        if value_type is None:
            value_type = self._resolve_var_dtype(value)
        if isinstance(value_type, numba_mlir_types.Boolean) or not isinstance(
            value_type, numba_mlir_types.Integer
        ):
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase '{op_name}' {parameter} must be an "
                "integer, not bool or a noninteger scalar"
            )


__all__ = ["_RadixRewrite"]
