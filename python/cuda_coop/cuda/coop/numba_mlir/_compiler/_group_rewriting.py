# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declared before-inference rewrite interface for primitive families."""

from __future__ import annotations

from typing import Any

from ._rewrite_support import _ThreadDataSpec, ir


class GroupRewriteContext:
    """Stable cross-family view of one before-inference rewrite."""

    __slots__ = ("__rewrite",)

    def __init__(self, rewrite: Any) -> None:
        self.__rewrite = rewrite

    def thread_data(self, value: ir.Var) -> _ThreadDataSpec | None:
        """Return the statically known ``ThreadData`` description."""

        return self.__rewrite._resolve_thread_data_spec(value)

    def is_typed_group_payload(self, value: ir.Var) -> bool:
        """Whether *value* originates from a typed group result."""

        return self.__rewrite._is_typed_group_payload_var(value)

    def array(self, value: ir.Var) -> _ThreadDataSpec | None:
        """Return statically known local/shared array dtype and extent facts."""

        return self.__rewrite._resolve_array_spec_from_var(value, seen=set())

    def dtype(self, value: ir.Var) -> Any | None:
        """Return the dtype inferred for an IR value, when known."""

        return self.__rewrite._resolve_var_dtype(value)

    def numba_type(self, value: ir.Var) -> Any | None:
        """Return the compiler type assigned to an IR value, when known."""

        return self.__rewrite._resolve_var_numba_type(value)

    def infer_thread_data_write_dtype(self, value: ir.Var) -> Any | None:
        """Infer an untyped ``ThreadData`` dtype from its element writes."""

        return self.__rewrite._infer_thread_data_dtype_from_writes(value)

    def record_thread_data_dtype(self, value: ir.Var, dtype: Any) -> None:
        """Record a dtype inferred for an otherwise untyped payload."""

        self.__rewrite._record_inferred_thread_data_dtype(value, dtype)

    def static_scalar_provenance(self, value: Any) -> Any:
        """Resolve a scalar only when it has explicitly static provenance."""

        return self.__rewrite._resolve_static_scalar_provenance(value)


__all__ = ["GroupRewriteContext"]
