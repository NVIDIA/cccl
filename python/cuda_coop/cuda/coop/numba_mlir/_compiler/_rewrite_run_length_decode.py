# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run Length Decode factory finalization.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_payload import PayloadInference
from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _dtype_values_match,
    _numba_types,
    ir,
    normalize_dtype_param,
)


class _RunLengthDecodeRewrite:
    def _infer_run_length_decode_payload(self, inference: PayloadInference) -> None:
        """Infer and cross-check the fused decode payloads."""

        with_relative_offsets = inference.factory_kwargs.get("with_relative_offsets")
        if not isinstance(with_relative_offsets, bool):
            raise CoopSinglePhaseRewriteError(
                "coop run_length_decode with_relative_offsets must be a "
                "compile-time bool"
            )
        expected_count = 6 if with_relative_offsets else 5
        if len(inference.runtime_args) != expected_count:
            raise CoopSinglePhaseRewriteError(
                "coop run_length_decode runtime arguments do not match "
                "with_relative_offsets"
            )

        run_values_var, run_values_spec = inference.array_candidate(0)
        run_lengths_var, run_lengths_spec = inference.array_candidate(1)
        total_var, total_spec = inference.array_candidate(2)
        decoded_var, decoded_spec = inference.array_candidate(3)
        relative_var = None
        relative_spec = None
        if with_relative_offsets:
            relative_var, relative_spec = inference.array_candidate(4)

        required_specs = {
            "run_values": run_values_spec,
            "run_lengths": run_lengths_spec,
            "total_decoded_size": total_spec,
            "decoded_items": decoded_spec,
        }
        if with_relative_offsets:
            required_specs["relative_offsets"] = relative_spec
        for name, spec in required_specs.items():
            if spec is None or spec.items_per_thread is None:
                raise CoopSinglePhaseRewriteError(
                    f"coop run_length_decode requires a fixed-size {name} array"
                )

        assert run_values_spec is not None
        assert run_lengths_spec is not None
        assert total_spec is not None
        assert decoded_spec is not None
        runs_per_thread = run_values_spec.items_per_thread
        if run_lengths_spec.items_per_thread != runs_per_thread:
            raise CoopSinglePhaseRewriteError(
                "coop run_length_decode run_values and run_lengths must "
                "have matching extents"
            )
        if total_spec.items_per_thread != 1:
            raise CoopSinglePhaseRewriteError(
                "coop run_length_decode total_decoded_size must contain "
                "exactly one item"
            )
        decoded_items_per_thread = inference.factory_value("decoded_items_per_thread")
        if decoded_spec.items_per_thread != decoded_items_per_thread:
            raise CoopSinglePhaseRewriteError(
                "coop run_length_decode decoded output extent must match "
                "decoded_items_per_thread"
            )
        if with_relative_offsets:
            assert relative_spec is not None
            if relative_spec.items_per_thread != decoded_items_per_thread:
                raise CoopSinglePhaseRewriteError(
                    "coop run_length_decode relative_offsets extent must "
                    "match decoded_items_per_thread"
                )

        item_dtype = inference.inferred_array_dtype(run_values_var, run_values_spec)
        decoded_dtype = inference.inferred_array_dtype(decoded_var, decoded_spec)
        if (
            item_dtype is not None
            and decoded_dtype is not None
            and not _dtype_values_match(item_dtype, decoded_dtype)
        ):
            raise CoopSinglePhaseRewriteError(
                "coop run_length_decode decoded dtype must match run_values"
            )
        item_dtype = item_dtype if item_dtype is not None else decoded_dtype
        run_length_dtype = inference.inferred_array_dtype(
            run_lengths_var, run_lengths_spec
        )
        total_dtype = inference.inferred_array_dtype(total_var, total_spec)
        if (
            run_length_dtype is not None
            and total_dtype is not None
            and not _dtype_values_match(run_length_dtype, total_dtype)
        ):
            raise CoopSinglePhaseRewriteError(
                "coop run_length_decode total_decoded_size dtype must match run_lengths"
            )
        total_dtype = total_dtype if total_dtype is not None else run_length_dtype
        decoded_offset_dtype = inference.factory_value("decoded_offset_dtype")
        if decoded_offset_dtype is None:
            decoded_offset_dtype = run_length_dtype
        if (
            run_length_dtype is not None
            and decoded_offset_dtype is not None
            and not _dtype_values_match(
                run_length_dtype,
                decoded_offset_dtype,
            )
        ):
            raise CoopSinglePhaseRewriteError(
                "coop run_length_decode decoded offset dtype must match run_lengths"
            )

        inference.infer_kwarg("runs_per_thread", runs_per_thread)
        inference.infer_kwarg("item_dtype", item_dtype)
        inference.infer_kwarg("run_length_dtype", run_length_dtype)
        inference.infer_kwarg("total_decoded_size_dtype", total_dtype)
        inference.infer_kwarg("decoded_offset_dtype", decoded_offset_dtype)
        if with_relative_offsets:
            relative_dtype = inference.inferred_array_dtype(relative_var, relative_spec)
            if (
                run_length_dtype is not None
                and relative_dtype is not None
                and not _dtype_values_match(run_length_dtype, relative_dtype)
            ):
                raise CoopSinglePhaseRewriteError(
                    "coop run_length_decode relative_offsets dtype must "
                    "match run_lengths"
                )
            inference.infer_kwarg("relative_offset_dtype", relative_dtype)

        for value, dtype in (
            (run_values_var, item_dtype),
            (decoded_var, item_dtype),
            (run_lengths_var, run_length_dtype),
            (total_var, run_length_dtype),
            (relative_var, run_length_dtype),
        ):
            if value is not None and dtype is not None:
                self._record_inferred_thread_data_dtype(value, dtype)

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
