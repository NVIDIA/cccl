# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Load and store payload inference for the Numba-CUDA-MLIR rewrite."""

from ._rewrite_payload import PayloadInference
from ._rewrite_support import ir


class _LoadStoreRewrite:
    def _infer_load_store_payload(self, inference: PayloadInference) -> None:
        payload_var, payload_spec = inference.candidate(1)
        if payload_spec is None:
            if inference.op_name in {"store", "warp_store"}:
                inferred_dtype = None
                for arg in inference.runtime_args[:2]:
                    if isinstance(arg, ir.Var):
                        inferred_dtype = self._resolve_var_dtype(arg)
                    if inferred_dtype is not None:
                        break
                inference.infer_kwarg("items_per_thread", 1)
                inference.infer_kwarg("dtype", inferred_dtype)
            return
        inference.infer_kwarg("items_per_thread", payload_spec.items_per_thread)
        inferred_dtype = payload_spec.dtype
        if (
            inferred_dtype is None
            and inference.runtime_args
            and isinstance(inference.runtime_args[0], ir.Var)
        ):
            inferred_dtype = self._resolve_var_dtype(inference.runtime_args[0])
        if inferred_dtype is None:
            inferred_dtype = inference.factory_value("dtype")
        inference.infer_kwarg("dtype", inferred_dtype)
        if inferred_dtype is not None and payload_var is not None:
            self._record_inferred_thread_data_dtype(payload_var, inferred_dtype)


__all__ = ["_LoadStoreRewrite"]
