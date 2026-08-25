# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scan callback and aggregate factory finalization.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_payload import PayloadInference
from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _dtype_values_match,
)


class _ScanRewrite:
    def _infer_scan_payload(self, inference: PayloadInference) -> None:
        """Infer block- or warp-scan payload dtype metadata."""

        if inference.op_name == "scan":
            input_var, input_spec = inference.candidate(0)
            output_var, output_spec = inference.candidate(1)
            self._require_matching_items_per_thread(
                inference.op_name, "input", input_spec, "output", output_spec
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
                inferred_dtype = inference.factory_value("dtype")
            extent = input_spec.items_per_thread if input_spec is not None else None
            if extent is None and output_spec is not None:
                extent = output_spec.items_per_thread
            inference.infer_kwarg("items_per_thread", extent)
            inference.infer_kwarg("dtype", inferred_dtype)
            for payload_var in (input_var, output_var):
                if inferred_dtype is not None and payload_var is not None:
                    self._record_inferred_thread_data_dtype(payload_var, inferred_dtype)
            if inference.factory_kwargs.get("block_aggregate"):
                aggregate_var, aggregate_spec = inference.candidate(2)
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

        value_var, value_spec = inference.candidate(0)
        inferred_dtype = value_spec.dtype if value_spec is not None else None
        if inferred_dtype is None and value_var is not None:
            inferred_dtype = self._resolve_var_dtype(value_var)
        if inferred_dtype is None:
            inferred_dtype = inference.factory_value("dtype")
        inference.infer_kwarg("dtype", inferred_dtype)
        aggregate_index = None
        if inference.factory_kwargs.get("warp_aggregate"):
            aggregate_index = (
                2
                if inference.factory_kwargs.get("valid_items")
                and inference.op_name in {"warp_exclusive_scan", "warp_inclusive_scan"}
                else 1
            )
        if aggregate_index is None:
            return
        aggregate_var, aggregate_spec = inference.candidate(aggregate_index)
        if aggregate_spec is None or aggregate_spec.items_per_thread != 1:
            raise CoopSinglePhaseRewriteError(
                "coop scan warp_aggregate must be a one-item ThreadData or local array."
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
            self._record_inferred_thread_data_dtype(aggregate_var, inferred_dtype)

    def _finalize_scan_factory_kwargs(
        self,
        *,
        runtime_arg_count: int,
        factory_kwargs: dict[str, object],
    ) -> None:
        from .._stateful_function import StatefulFunction

        prefix_callback = factory_kwargs.get("block_prefix_callback_op")
        if prefix_callback is None:
            prefix_callback = factory_kwargs.get("prefix_op")
        stateful = isinstance(prefix_callback, StatefulFunction)
        expected_count = 3 if stateful else 2
        if runtime_arg_count != expected_count:
            requirement = (
                "requires a third runtime state argument"
                if stateful
                else "accepts only input and output runtime arguments"
            )
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase 'scan' {requirement}."
            )


__all__ = ["_ScanRewrite"]
