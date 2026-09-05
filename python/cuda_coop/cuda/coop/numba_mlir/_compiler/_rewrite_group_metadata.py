# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Group-root metadata and store-mutation analysis.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    ir,
)


class _GroupMetadataRewrite:
    @staticmethod
    def _store_algorithm_mutates_payload(op_name: str, algorithm: object) -> bool:
        if isinstance(algorithm, bool):
            return False
        if isinstance(algorithm, int):
            from .._enums import BlockStoreAlgorithm, WarpStoreAlgorithm

            enum_type = (
                WarpStoreAlgorithm if op_name == "warp_store" else BlockStoreAlgorithm
            )
            try:
                resolved = enum_type(algorithm)
            except ValueError:
                return False
            if op_name == "warp_store":
                return resolved is WarpStoreAlgorithm.TRANSPOSE
            return resolved in {
                BlockStoreAlgorithm.TRANSPOSE,
                BlockStoreAlgorithm.WARP_TRANSPOSE,
                BlockStoreAlgorithm.WARP_TRANSPOSE_TIMESLICED,
            }
        normalized = str(algorithm).lower()
        if op_name == "warp_store":
            return normalized in {"transpose", "::cub::warp_store_transpose"}
        return normalized in {
            "transpose",
            "warp_transpose",
            "warp_transpose_timesliced",
            "::cub::block_store_transpose",
            "::cub::block_store_warp_transpose",
            "::cub::block_store_warp_transpose_timesliced",
        }

    def _extract_group_root_match_metadata(
        self,
        *,
        op_name: str,
        runtime_args: tuple[ir.Var, ...],
        factory_kwargs: dict[str, object],
    ) -> tuple[bool, bool, bool]:
        physical_warp_tile_origin = factory_kwargs.pop(
            "_physical_warp_tile_origin", False
        )
        group_root_store = factory_kwargs.pop("_group_root_store", False)
        common_root_operation = factory_kwargs.pop("_common_root_operation", None)
        if not isinstance(physical_warp_tile_origin, bool):
            raise CoopSinglePhaseRewriteError(
                "_physical_warp_tile_origin must be a compile-time bool"
            )
        if not isinstance(group_root_store, bool):
            raise CoopSinglePhaseRewriteError(
                "_group_root_store must be a compile-time bool"
            )
        if common_root_operation is not None:
            operation_families = {
                "load": frozenset({"load"}),
                "warp_load": frozenset({"load"}),
                "store": frozenset({"store"}),
                "warp_store": frozenset({"store"}),
                "group_reduce": frozenset({"reduce", "sum"}),
                "block_reduce_builtin": frozenset({"reduce", "sum"}),
                "reduce": frozenset({"reduce"}),
                "sum": frozenset({"reduce", "sum"}),
                "warp_reduce_builtin": frozenset({"reduce", "sum"}),
                "warp_reduce": frozenset({"reduce"}),
                "warp_sum": frozenset({"reduce", "sum"}),
                "scan": frozenset(
                    {
                        "scan",
                        "exclusive_sum",
                        "inclusive_sum",
                        "exclusive_scan",
                        "inclusive_scan",
                    }
                ),
                "warp_exclusive_sum": frozenset({"scan", "exclusive_sum"}),
                "warp_inclusive_sum": frozenset({"scan", "inclusive_sum"}),
                "warp_exclusive_scan": frozenset({"scan", "exclusive_scan"}),
                "warp_inclusive_scan": frozenset({"scan", "inclusive_scan"}),
                "exchange": frozenset({"exchange"}),
                "warp_exchange": frozenset({"exchange"}),
                "adjacent_difference": frozenset({"adjacent_difference"}),
                "discontinuity": frozenset({"discontinuity"}),
                "shuffle": frozenset({"shuffle"}),
                "_group_histogram": frozenset({"histogram"}),
                "_group_run_length_decode": frozenset({"run_length_decode"}),
            }
            if common_root_operation not in operation_families.get(
                op_name, frozenset()
            ):
                raise CoopSinglePhaseRewriteError(
                    "_common_root_operation does not match the rewritten group operation"
                )
            from ._parameters import (
                _validate_common_histogram_dtypes,
                _validate_common_numeric_dtype,
                _validate_common_run_length_decode_dtypes,
            )

            if op_name == "_group_histogram":
                try:
                    _validate_common_histogram_dtypes(
                        factory_kwargs.get("item_dtype"),
                        factory_kwargs.get("counter_dtype"),
                    )
                except (TypeError, ValueError) as exc:
                    raise CoopSinglePhaseRewriteError(str(exc)) from exc
            elif op_name == "_group_run_length_decode":
                try:
                    _validate_common_run_length_decode_dtypes(
                        factory_kwargs.get("item_dtype"),
                        factory_kwargs.get("run_length_dtype"),
                    )
                except (TypeError, ValueError) as exc:
                    raise CoopSinglePhaseRewriteError(str(exc)) from exc
            elif op_name in {"load", "warp_load", "store", "warp_store"}:
                operand_names = (
                    ("source", "output")
                    if op_name in {"load", "warp_load"}
                    else ("destination", "value")
                )
                for operand_name, operand in zip(operand_names, runtime_args):
                    operand_dtype = self._resolve_var_dtype(operand)
                    if operand_dtype is None:
                        raise CoopSinglePhaseRewriteError(
                            f"Failed to infer cuda.coop.{common_root_operation} {operand_name} dtype for portable API validation."
                        )
                    try:
                        _validate_common_numeric_dtype(
                            operand_dtype, operation=common_root_operation
                        )
                    except (TypeError, ValueError) as exc:
                        raise CoopSinglePhaseRewriteError(str(exc)) from exc
            else:
                try:
                    _validate_common_numeric_dtype(
                        factory_kwargs.get("dtype"), operation=common_root_operation
                    )
                except (TypeError, ValueError) as exc:
                    raise CoopSinglePhaseRewriteError(str(exc)) from exc
        root_store_scalar = False
        preserve_root_store_payload = False
        if group_root_store:
            if op_name not in {"store", "warp_store"} or len(runtime_args) < 2:
                raise CoopSinglePhaseRewriteError(
                    "_group_root_store is valid only for root store calls"
                )
            root_store_scalar = self._resolve_thread_data_spec(runtime_args[1]) is None
            preserve_root_store_payload = (
                root_store_scalar
                or self._store_algorithm_mutates_payload(
                    op_name, factory_kwargs.get("algorithm", "direct")
                )
            )
        return (
            physical_warp_tile_origin,
            preserve_root_store_payload,
            root_store_scalar,
        )


__all__ = ["_GroupMetadataRewrite"]
