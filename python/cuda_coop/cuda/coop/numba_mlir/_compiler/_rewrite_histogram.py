# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Histogram payload inference for the Numba-CUDA-MLIR rewrite."""

from ._rewrite_payload import PayloadInference
from ._rewrite_support import CoopSinglePhaseRewriteError


class _HistogramRewrite:
    def _infer_histogram_payload(self, inference: PayloadInference) -> None:
        samples_var, samples_spec = inference.array_candidate(0)
        histogram_var, histogram_spec = inference.array_candidate(1)
        if samples_spec is None or samples_spec.items_per_thread is None:
            raise CoopSinglePhaseRewriteError(
                "coop histogram requires a fixed-size samples array"
            )
        if histogram_spec is None or histogram_spec.items_per_thread is None:
            raise CoopSinglePhaseRewriteError(
                "coop histogram requires a fixed-size counter array"
            )
        inference.infer_kwarg("items_per_thread", samples_spec.items_per_thread)
        inference.infer_kwarg("bins", histogram_spec.items_per_thread)
        item_dtype = inference.inferred_array_dtype(samples_var, samples_spec)
        counter_dtype = inference.inferred_array_dtype(histogram_var, histogram_spec)
        inference.infer_kwarg("item_dtype", item_dtype)
        inference.infer_kwarg("counter_dtype", counter_dtype)
        if item_dtype is not None and samples_var is not None:
            self._record_inferred_thread_data_dtype(samples_var, item_dtype)


__all__ = ["_HistogramRewrite"]
