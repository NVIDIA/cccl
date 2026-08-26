# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared payload-inference mechanics and family dispatch.

Primitive-specific inference lives in the matching ``_rewrite_<family>``
mixin. This module owns only the common inference context and dispatch order.
"""

from typing import Any

from ._rewrite_support import (
    CoopSinglePhaseRewriteError,
    _ThreadDataSpec,
    ir,
    normalize_dtype_param,
)


class PayloadInference:
    """Mutable context shared by the primitive-specific inference handlers."""

    _COMMON_OPERATION_ALIASES: dict[str, str] = {}
    _OPERATION_ALIASES: dict[str, str] = {}
    _DTYPE_FACTORY_KWARGS = frozenset({"dtype"})

    def __init__(
        self,
        rewrite: Any,
        op_name: str,
        runtime_args: list[ir.Var],
        allowed_factory_kwargs: set[str],
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        self.rewrite = rewrite
        self.op_name = self._OPERATION_ALIASES.get(op_name, op_name)
        self.portable_op_name = self._COMMON_OPERATION_ALIASES.get(op_name)
        self.runtime_args = runtime_args
        self.allowed_factory_kwargs = allowed_factory_kwargs
        self.seen_factory_kwargs = seen_factory_kwargs
        self.factory_kwargs = factory_kwargs

    def factory_value(self, name: str):
        return self.factory_kwargs.get(name)

    def _factory_kwarg_matches(self, name: str, actual, expected) -> bool:
        if name in self._DTYPE_FACTORY_KWARGS:
            try:
                actual = normalize_dtype_param(actual)
                expected = normalize_dtype_param(expected)
            except (TypeError, ValueError):
                pass
        return actual == expected

    def infer_kwarg(self, name: str, value) -> None:
        if name not in self.allowed_factory_kwargs or value is None:
            return
        if name in self.seen_factory_kwargs:
            if not self._factory_kwarg_matches(name, self.factory_kwargs[name], value):
                raise CoopSinglePhaseRewriteError(
                    f"coop movement {self.op_name!r} {name} does not match "
                    "the value inferred from ThreadData."
                )
            return
        self.factory_kwargs[name] = value
        self.seen_factory_kwargs.add(name)

    def candidate(self, index: int) -> tuple[ir.Var | None, _ThreadDataSpec | None]:
        if not 0 <= index < len(self.runtime_args):
            return (None, None)
        value = self.runtime_args[index]
        if not isinstance(value, ir.Var):
            return (None, None)
        spec = self.rewrite._resolve_thread_data_spec(value)
        if self.rewrite._is_typed_group_payload_var(value) and (
            spec is None or spec.items_per_thread is None
        ):
            raise CoopSinglePhaseRewriteError(
                f"coop movement {self.op_name!r} could not infer the static "
                "extent of a typed group payload"
            )
        return (value, spec)

    def array_candidate(
        self, index: int
    ) -> tuple[ir.Var | None, _ThreadDataSpec | None]:
        if not 0 <= index < len(self.runtime_args):
            return (None, None)
        value = self.runtime_args[index]
        if not isinstance(value, ir.Var):
            return (None, None)
        spec = self.rewrite._resolve_array_spec_from_var(value, seen=set())
        if self.rewrite._is_typed_group_payload_var(value) and (
            spec is None or spec.items_per_thread is None
        ):
            raise CoopSinglePhaseRewriteError(
                f"coop movement {self.op_name!r} could not infer the static "
                "extent of a typed group payload"
            )
        return (value, spec)

    def inferred_array_dtype(
        self,
        value: ir.Var | None,
        spec: _ThreadDataSpec | None,
    ):
        dtype = spec.dtype if spec is not None else None
        if dtype is None and value is not None:
            dtype = self.rewrite._resolve_var_dtype(value)
        if dtype is None and value is not None:
            dtype = self.rewrite._infer_thread_data_dtype_from_writes(value)
        return dtype


class _PayloadRewrite:
    """Dispatch payload inference to the owning primitive-family mixin."""

    def _infer_factory_kwargs_from_thread_data(
        self,
        op_name: str,
        runtime_args: list[ir.Var],
        allowed_factory_kwargs: set[str],
        seen_factory_kwargs: set[str],
        factory_kwargs: dict[str, object],
    ) -> None:
        inference = PayloadInference(
            self,
            op_name,
            runtime_args,
            allowed_factory_kwargs,
            seen_factory_kwargs,
            factory_kwargs,
        )
        operation = inference.op_name

        if operation in {
            "group_reduce",
            "block_reduce_builtin",
            "reduce",
            "sum",
            "warp_reduce_builtin",
            "warp_reduce",
            "warp_sum",
        }:
            self._infer_reduce_payload(inference)
        elif operation in {
            "scan",
            "warp_exclusive_sum",
            "warp_inclusive_sum",
            "warp_exclusive_scan",
            "warp_inclusive_scan",
        }:
            self._infer_scan_payload(inference)
        elif operation in {"load", "store", "warp_load", "warp_store"}:
            self._infer_load_store_payload(inference)


__all__ = ["_PayloadRewrite"]
