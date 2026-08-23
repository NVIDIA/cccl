# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Merge Sort callable and sentinel validation.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    _UNRESOLVED,
    CoopSinglePhaseRewriteError,
    _numba_types,
    ir,
    normalize_dtype_param,
    np,
    operator,
)


class _MergeSortRewrite:
    @staticmethod
    def _lossless_merge_sort_sentinel(value: object, key_dtype: object) -> object:
        from ._parameters import _NUMBA_MLIR_DTYPE_NAMES

        try:
            key_dtype = normalize_dtype_param(key_dtype)
        except (TypeError, ValueError):
            return value
        dtype_name = _NUMBA_MLIR_DTYPE_NAMES.get(key_dtype)
        if dtype_name is None:
            return value
        numpy_dtype = np.dtype(dtype_name)

        if key_dtype == _numba_types.boolean:
            if not isinstance(value, (bool, np.bool_)):
                raise CoopSinglePhaseRewriteError(
                    "Merge Sort oob_default must have the same bool dtype as keys"
                )
            return np.bool_(value)

        if isinstance(key_dtype, _numba_types.Integer):
            if isinstance(value, (bool, np.bool_)):
                raise CoopSinglePhaseRewriteError(
                    "Merge Sort oob_default must be an integer, not bool"
                )
            try:
                integer = operator.index(value)
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(
                    "Merge Sort oob_default must be an integer for integer keys"
                ) from exc
            limits = np.iinfo(numpy_dtype)
            if not limits.min <= integer <= limits.max:
                raise CoopSinglePhaseRewriteError(
                    f"Merge Sort oob_default={integer} is not representable "
                    f"in keys dtype {dtype_name}"
                )
            return numpy_dtype.type(integer)

        if isinstance(key_dtype, _numba_types.Float):
            if isinstance(value, (bool, np.bool_)):
                raise CoopSinglePhaseRewriteError(
                    "Merge Sort oob_default must be numeric, not bool"
                )
            if not isinstance(value, (int, float, np.integer, np.floating)):
                raise CoopSinglePhaseRewriteError(
                    "Merge Sort oob_default must be numeric for floating keys"
                )
            with np.errstate(over="ignore", invalid="ignore"):
                converted = numpy_dtype.type(value)
            original_float = float(value)
            converted_float = float(converted)
            exact = original_float == converted_float
            if exact and original_float == 0.0:
                exact = np.signbit(original_float) == np.signbit(converted_float)
            if isinstance(value, (int, np.integer)) and np.isfinite(converted_float):
                exact = exact and int(converted_float) == int(value)
            if not exact or np.isnan(original_float):
                raise CoopSinglePhaseRewriteError(
                    f"Merge Sort oob_default={value!r} is not losslessly "
                    f"representable in keys dtype {dtype_name}"
                )
            return converted

        return value

    def _validate_merge_sort_runtime_controls(
        self,
        *,
        op_name: str,
        runtime_args: list[ir.Var],
        control_vars: dict[str, ir.Var],
        factory_kwargs: dict[str, object],
    ) -> tuple[tuple[int, object], ...]:
        if op_name not in {
            "merge_sort_keys",
            "merge_sort_pairs",
            "warp_merge_sort_keys",
            "warp_merge_sort_pairs",
        }:
            return ()

        operation = "merge_sort_pairs" if "pairs" in op_name else "merge_sort_keys"
        prefix = f"cuda.coop.numba_mlir.{operation}"
        valid_items_var = control_vars.get("valid_items")
        if valid_items_var is not None:
            static_valid_items = self._resolve_factory_kwarg_value(
                "valid_items", valid_items_var
            )
            if static_valid_items is not _UNRESOLVED:
                if isinstance(static_valid_items, (bool, np.bool_)):
                    raise CoopSinglePhaseRewriteError(
                        f"{prefix} valid_items must be an integer, not bool"
                    )
                try:
                    operator.index(static_valid_items)
                except TypeError as exc:
                    raise CoopSinglePhaseRewriteError(
                        f"{prefix} valid_items must be an integer"
                    ) from exc
            else:
                valid_items_dtype = self._resolve_var_dtype(valid_items_var)
                if valid_items_dtype is None:
                    valid_items_dtype = self._resolve_var_numba_type(valid_items_var)
                if valid_items_dtype == _numba_types.boolean:
                    raise CoopSinglePhaseRewriteError(
                        f"{prefix} valid_items must be an integer, not bool"
                    )
                if valid_items_dtype is not None and not isinstance(
                    valid_items_dtype, _numba_types.Integer
                ):
                    raise CoopSinglePhaseRewriteError(
                        f"{prefix} valid_items must have an integer dtype"
                    )

        oob_default_var = control_vars.get("oob_default")
        if oob_default_var is None:
            return ()
        key_dtype = factory_kwargs.get("keys" if "pairs" in op_name else "dtype")
        if key_dtype is None:
            raise CoopSinglePhaseRewriteError(
                f"{prefix} could not infer the keys dtype before validating oob_default"
            )
        static_oob_default = self._resolve_factory_kwarg_value(
            "oob_default", oob_default_var
        )
        if static_oob_default is not _UNRESOLVED:
            converted = self._lossless_merge_sort_sentinel(
                static_oob_default,
                key_dtype,
            )
            argument_index = next(
                index
                for index, argument in enumerate(runtime_args)
                if argument is oob_default_var or argument.name == oob_default_var.name
            )
            return ((argument_index, converted),)

        oob_default_dtype = self._resolve_var_dtype(oob_default_var)
        if oob_default_dtype is None:
            oob_default_dtype = self._resolve_var_numba_type(oob_default_var)
        if oob_default_dtype is None:
            return ()
        try:
            key_dtype = normalize_dtype_param(key_dtype)
            oob_default_dtype = normalize_dtype_param(oob_default_dtype)
        except (TypeError, ValueError):
            pass
        if oob_default_dtype != key_dtype:
            raise CoopSinglePhaseRewriteError(
                f"{prefix} oob_default must have the same dtype as keys "
                f"({key_dtype}); got {oob_default_dtype}"
            )
        return ()


__all__ = ["_MergeSortRewrite"]
