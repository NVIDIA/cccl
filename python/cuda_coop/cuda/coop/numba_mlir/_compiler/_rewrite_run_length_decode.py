# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run Length Decode factory finalization.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _dtype_values_match,
    _numba_types,
    ir,
    normalize_dtype_param,
)


class _RunLengthDecodeRewrite:
    def _finalize_group_run_length_decode_factory_kwargs(
        self,
        *,
        runtime_args: list[ir.Var],
        factory_kwargs: dict[str, object],
    ) -> None:
        """Validate the fused decode offset and remove planner metadata."""

        run_length_dtype = factory_kwargs.get("run_length_dtype")
        try:
            run_length_dtype = normalize_dtype_param(run_length_dtype)
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(
                "coop run_length_decode could not resolve run_lengths dtype"
            ) from exc
        if isinstance(run_length_dtype, _numba_types.Boolean) or not isinstance(
            run_length_dtype,
            _numba_types.Integer,
        ):
            raise CoopSinglePhaseRewriteError(
                "coop run_length_decode run_lengths must have an integer dtype"
            )

        static_offset = factory_kwargs.pop(
            "_static_decoded_window_offset",
            None,
        )
        if static_offset is not None:
            if isinstance(static_offset, bool) or not isinstance(static_offset, int):
                raise CoopSinglePhaseRewriteError(
                    "coop run_length_decode decoded_window_offset must be an integer"
                )
            if static_offset < 0:
                raise CoopSinglePhaseRewriteError(
                    "coop run_length_decode decoded_window_offset must be non-negative"
                )
            bitwidth = getattr(run_length_dtype, "bitwidth", None)
            signed = getattr(run_length_dtype, "signed", None)
            if isinstance(bitwidth, int) and isinstance(signed, bool):
                value_bits = bitwidth - 1 if signed else bitwidth
                if static_offset >= 1 << value_bits:
                    raise CoopSinglePhaseRewriteError(
                        "coop run_length_decode decoded_window_offset must be "
                        "representable in the run_lengths dtype"
                    )
            return

        offset_var = runtime_args[-1]
        offset_dtype = self._resolve_var_dtype(offset_var)
        if offset_dtype is None:
            offset_dtype = self._resolve_var_numba_type(offset_var)
        try:
            offset_dtype = normalize_dtype_param(offset_dtype)
        except (TypeError, ValueError) as exc:
            raise CoopSinglePhaseRewriteError(
                "coop run_length_decode decoded_window_offset must have an "
                "integer dtype"
            ) from exc
        if isinstance(offset_dtype, _numba_types.Boolean) or not isinstance(
            offset_dtype,
            _numba_types.Integer,
        ):
            raise CoopSinglePhaseRewriteError(
                "coop run_length_decode decoded_window_offset must have an "
                "integer dtype"
            )
        if not _dtype_values_match(offset_dtype, run_length_dtype):
            raise CoopSinglePhaseRewriteError(
                "coop run_length_decode decoded_window_offset dtype must match "
                "run_lengths"
            )


__all__ = ["_RunLengthDecodeRewrite"]
