# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose payload and scalar facts to operation-specific call rewriting.

After group resolution has selected implementations,
``CoopSinglePhaseRewrite`` uses primitive-specific hooks to infer factory
arguments and prepare runtime operands. ``GroupRewriteContext`` gives those
hooks access to the active rewrite's array descriptions, compiler types, and
recorded ``ThreadData`` element types. It forwards queries and updates to the
rewrite so all operations use the same analysis and caches.

This context is used during the second phase of ``CoopWholeFunctionPlanner``.
It does not register a separate Numba rewrite or own the IR; its returned
facts include only what the active helper knows before type inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._rewrite_support import _ThreadDataSpecification, ir

if TYPE_CHECKING:
    from ._rewrite import CoopSinglePhaseRewrite


class GroupRewriteContext:
    """Share the active call rewrite's analysis with primitive-specific hooks.

    Parameters
    ----------
    rewrite : CoopSinglePhaseRewrite
        Active rewrite supplying IR facts to primitive-family hooks.
    """

    __slots__ = ("__rewrite",)

    def __init__(self, rewrite: CoopSinglePhaseRewrite) -> None:
        self.__rewrite = rewrite

    def thread_data(self, value: ir.Var) -> _ThreadDataSpecification | None:
        """Return known payload facts, or ``None`` when none can be recovered.

        A description can have an unknown dtype or extent. It can also
        describe a native local array; it does not prove that the value came
        from public ``ThreadData``.
        """

        return self.__rewrite._resolve_thread_data_specification(value)

    def is_typed_group_payload(self, value: ir.Var) -> bool:
        """Check for a result marker among the variable's definitions.

        Inference uses this to reject a generated result whose extent remains
        unknown, rather than silently treating it as an unrelated operand.
        """

        return self.__rewrite._is_typed_group_payload_var(value)

    def array(self, value: ir.Var) -> _ThreadDataSpecification | None:
        """Return known dtype and extent facts for local or shared arrays."""

        return self.__rewrite._resolve_array_specification_from_var(
            value, seen=set()
        )

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
        """Resolve a static scalar with its known source dtype.

        Return ``_UNRESOLVED`` when the analysis cannot establish static
        origin. A provenance record with value ``None`` remains valid.
        """

        return self.__rewrite._resolve_static_scalar_provenance(value)


__all__ = ["GroupRewriteContext"]
