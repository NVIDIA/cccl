# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""TopK runtime-control validation.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    _UNRESOLVED,
    CoopSinglePhaseRewriteError,
    ir,
    normalize_dim_param,
    np,
    operator,
)


class _TopKRewrite:
    def _validate_topk_runtime_controls(
        self,
        *,
        op_name: str,
        runtime_args: list[ir.Var],
        factory_kwargs: dict[str, object],
    ) -> None:
        operation = {
            "topk_max_keys": "topk_max_keys",
            "topk_max_pairs": "topk_max_pairs",
            "topk_min_keys": "topk_min_keys",
            "topk_min_pairs": "topk_min_pairs",
            "_common_topk_max_keys": "topk_max_keys",
            "_common_topk_max_pairs": "topk_max_pairs",
            "_common_topk_min_keys": "topk_min_keys",
            "_common_topk_min_pairs": "topk_min_pairs",
            "_qualified_group_topk_max_keys": "topk_max_keys",
            "_qualified_group_topk_max_pairs": "topk_max_pairs",
            "_qualified_group_topk_min_keys": "topk_min_keys",
            "_qualified_group_topk_min_pairs": "topk_min_pairs",
        }.get(op_name)
        if operation is None:
            return

        prefix = (
            f"cuda.coop.{operation}"
            if op_name.startswith("_common_")
            else f"cuda.coop.numba_mlir.{operation}"
        )
        base_count = 3 if operation.endswith("_pairs") else 2
        controls: dict[str, ir.Var] = {"k": runtime_args[base_count - 1]}
        control_index = base_count
        for name in ("num_valid", "begin_bit", "end_bit"):
            if factory_kwargs.get(name) is not True:
                continue
            if control_index >= len(runtime_args):
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} is missing runtime control {name}"
                )
            controls[name] = runtime_args[control_index]
            control_index += 1

        from numba_cuda_mlir import types as numba_mlir_types

        for name, value_var in controls.items():
            value_type = self._resolve_var_numba_type(value_var)
            if value_type is None:
                value_type = self._resolve_var_dtype(value_var)
            if isinstance(value_type, numba_mlir_types.Boolean) or (
                value_type is not None
                and not isinstance(value_type, numba_mlir_types.Integer)
            ):
                public_name = "valid_items" if name == "num_valid" else name
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} {public_name} must have an integer dtype"
                )

        def static_index(name: str) -> int | None:
            value_var = controls.get(name)
            if value_var is None:
                return None
            value = self._resolve_factory_kwarg_value(name, value_var)
            if value is _UNRESOLVED:
                return None
            if isinstance(value, (bool, np.bool_)):
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} {name} must be an int-like scalar"
                )
            try:
                normalized = operator.index(value)
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} {name} must be an int-like scalar"
                ) from exc
            if isinstance(normalized, bool):
                raise CoopSinglePhaseRewriteError(
                    f"{prefix} {name} must be an int-like scalar"
                )
            return int(normalized)

        static_k = static_index("k")
        if static_k is not None and static_k <= 0:
            raise CoopSinglePhaseRewriteError(f"{prefix} k must be positive")

        threads_per_block = factory_kwargs.get("threads_per_block")
        items_per_thread = factory_kwargs.get("items_per_thread")
        tile_size = None
        if threads_per_block is not None and isinstance(items_per_thread, int):
            dim = normalize_dim_param(threads_per_block)
            tile_size = dim.x * dim.y * dim.z * items_per_thread
        static_valid = (
            tile_size if "num_valid" not in controls else static_index("num_valid")
        )
        if static_valid is not None and (
            static_valid <= 0 or (tile_size is not None and static_valid > tile_size)
        ):
            raise CoopSinglePhaseRewriteError(
                f"{prefix} valid_items must be in [1, {tile_size}]"
            )
        if (
            static_k is not None
            and static_valid is not None
            and static_k > static_valid
        ):
            raise CoopSinglePhaseRewriteError(f"{prefix} k must be <= valid_items")

        key_dtype = factory_kwargs.get("dtype")
        if key_dtype is None:
            key_dtype = factory_kwargs.get("keys")
        key_width = getattr(key_dtype, "bitwidth", None)
        if key_width is None:
            return
        key_width = int(key_width)
        static_begin = 0 if "begin_bit" not in controls else static_index("begin_bit")
        static_end = key_width if "end_bit" not in controls else static_index("end_bit")
        if static_begin is not None and not 0 <= static_begin < key_width:
            raise CoopSinglePhaseRewriteError(
                f"{prefix} begin_bit must be in [0, {key_width})"
            )
        if static_end is not None and not 0 < static_end <= key_width:
            raise CoopSinglePhaseRewriteError(
                f"{prefix} end_bit must be in (0, {key_width}]"
            )
        if (
            static_begin is not None
            and static_end is not None
            and static_end <= static_begin
        ):
            raise CoopSinglePhaseRewriteError(f"{prefix} end_bit must exceed begin_bit")


__all__ = ["_TopKRewrite"]
