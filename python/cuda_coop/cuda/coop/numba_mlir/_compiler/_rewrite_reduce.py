# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reduction payload inference for the Numba-CUDA-MLIR rewrite."""

from ._rewrite_payload import PayloadInference


class _ReduceRewrite:
    def _infer_reduce_payload(self, inference: PayloadInference) -> None:
        payload_var, payload_spec = inference.candidate(0)
        if payload_spec is not None:
            inference.infer_kwarg("items_per_thread", payload_spec.items_per_thread)
        inferred_dtype = payload_spec.dtype if payload_spec is not None else None
        if inferred_dtype is None and payload_var is not None:
            inferred_dtype = self._resolve_var_dtype(payload_var)
        if inferred_dtype is None:
            inferred_dtype = inference.factory_value("dtype")
        inference.infer_kwarg("dtype", inferred_dtype)
        if inferred_dtype is not None and payload_var is not None:
            self._record_inferred_thread_data_dtype(payload_var, inferred_dtype)


__all__ = ["_ReduceRewrite"]
