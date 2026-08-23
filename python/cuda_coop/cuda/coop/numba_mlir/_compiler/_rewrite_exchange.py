# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block and warp exchange factory finalization.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _dtype_values_match,
    ir,
)


class _ExchangeRewrite:
    def _finalize_exchange_factory_kwargs(
        self,
        runtime_args: list[ir.Var],
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from cuda.coop.numba_mlir._lowering._exchange import (
            BlockExchangeType,
            _normalize_block_exchange_type,
        )

        try:
            exchange_type = _normalize_block_exchange_type(
                factory_kwargs.get(
                    "block_exchange_type", BlockExchangeType.StripedToBlocked
                )
            )
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(str(exc)) from exc
        uses_ranks = exchange_type in {
            BlockExchangeType.ScatterToBlocked,
            BlockExchangeType.ScatterToStriped,
            BlockExchangeType.ScatterToStripedGuarded,
            BlockExchangeType.ScatterToStripedFlagged,
        }
        uses_valid_flags = exchange_type == BlockExchangeType.ScatterToStripedFlagged
        if uses_valid_flags:
            expected_counts = {3, 4}
        elif uses_ranks:
            expected_counts = {2, 3}
        else:
            expected_counts = {1, 2}
        if runtime_arg_count not in expected_counts:
            expected_csv = ", ".join((str(v) for v in sorted(expected_counts)))
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase 'exchange' runtime argument count {runtime_arg_count} is incompatible with block_exchange_type={exchange_type.name}; expected one of {{{expected_csv}}}."
            )
        out_of_place = runtime_arg_count in {2, 3, 4} and (
            not uses_ranks
            and runtime_arg_count == 2
            or (uses_ranks and (not uses_valid_flags) and (runtime_arg_count == 3))
            or (uses_valid_flags and runtime_arg_count == 4)
        )
        if "use_output_items" in seen_factory_kwargs:
            requested_value_form = factory_kwargs["use_output_items"]
            if requested_value_form is not None and (
                not isinstance(requested_value_form, bool)
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'exchange' use_output_items must be a boolean or None."
                )
            if (
                requested_value_form is not None
                and requested_value_form != out_of_place
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'exchange' use_output_items does not match the runtime argument form."
                )
        factory_kwargs["use_output_items"] = out_of_place
        seen_factory_kwargs.add("use_output_items")
        ranks_idx = 2 if out_of_place else 1
        valid_flags_idx = 3 if out_of_place else 2
        if uses_ranks:
            ranks_var = runtime_args[ranks_idx]
            if not isinstance(ranks_var, ir.Var):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'exchange' ranks runtime argument must be a variable."
                )
            inferred_offset_dtype = self._resolve_var_dtype(ranks_var)
            if inferred_offset_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer offset_dtype from exchange ranks argument."
                )
            if "offset_dtype" in seen_factory_kwargs:
                if not _dtype_values_match(
                    factory_kwargs["offset_dtype"], inferred_offset_dtype
                ):
                    raise CoopSinglePhaseRewriteError(
                        "exchange offset_dtype does not match ranks argument dtype."
                    )
            else:
                factory_kwargs["offset_dtype"] = inferred_offset_dtype
                seen_factory_kwargs.add("offset_dtype")
        if uses_valid_flags:
            valid_flags_var = runtime_args[valid_flags_idx]
            if not isinstance(valid_flags_var, ir.Var):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'exchange' valid_flags runtime argument must be a variable."
                )
            inferred_valid_flag_dtype = self._resolve_var_dtype(valid_flags_var)
            if inferred_valid_flag_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer valid_flag_dtype from exchange valid_flags argument."
                )
            if "valid_flag_dtype" in seen_factory_kwargs:
                if not _dtype_values_match(
                    factory_kwargs["valid_flag_dtype"], inferred_valid_flag_dtype
                ):
                    raise CoopSinglePhaseRewriteError(
                        "exchange valid_flag_dtype does not match valid_flags argument dtype."
                    )
            else:
                factory_kwargs["valid_flag_dtype"] = inferred_valid_flag_dtype
                seen_factory_kwargs.add("valid_flag_dtype")

    def _finalize_warp_exchange_factory_kwargs(
        self,
        runtime_args: list[ir.Var],
        runtime_arg_count: int,
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        from cuda.coop.numba_mlir._lowering._exchange import (
            WarpExchangeType,
            _normalize_warp_exchange_type,
        )

        try:
            exchange_type = _normalize_warp_exchange_type(
                factory_kwargs.get(
                    "warp_exchange_type", WarpExchangeType.StripedToBlocked
                )
            )
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(str(exc)) from exc
        uses_ranks = exchange_type == WarpExchangeType.ScatterToStriped
        expected_counts = {2, 3} if uses_ranks else {2}
        if runtime_arg_count not in expected_counts:
            expected_csv = ", ".join((str(v) for v in sorted(expected_counts)))
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase 'warp_exchange' runtime argument count {runtime_arg_count} is incompatible with warp_exchange_type={exchange_type.name}; expected one of {{{expected_csv}}}."
            )
        if uses_ranks:
            inferred_use_output_items = runtime_arg_count == 3
            if "use_output_items" in seen_factory_kwargs:
                if factory_kwargs["use_output_items"] != inferred_use_output_items:
                    raise CoopSinglePhaseRewriteError(
                        "warp_exchange use_output_items does not match the runtime argument count."
                    )
            else:
                factory_kwargs["use_output_items"] = inferred_use_output_items
                seen_factory_kwargs.add("use_output_items")
            ranks_var = runtime_args[2 if inferred_use_output_items else 1]
            if not isinstance(ranks_var, ir.Var):
                raise CoopSinglePhaseRewriteError(
                    "coop single-phase 'warp_exchange' ranks runtime argument must be a variable."
                )
            inferred_offset_dtype = self._resolve_var_dtype(ranks_var)
            if inferred_offset_dtype is None:
                raise CoopSinglePhaseRewriteError(
                    "Failed to infer offset_dtype from warp_exchange ranks argument."
                )
            if "offset_dtype" in seen_factory_kwargs:
                if not _dtype_values_match(
                    factory_kwargs["offset_dtype"], inferred_offset_dtype
                ):
                    raise CoopSinglePhaseRewriteError(
                        "warp_exchange offset_dtype does not match ranks argument dtype."
                    )
            else:
                factory_kwargs["offset_dtype"] = inferred_offset_dtype
                seen_factory_kwargs.add("offset_dtype")


__all__ = ["_ExchangeRewrite"]
