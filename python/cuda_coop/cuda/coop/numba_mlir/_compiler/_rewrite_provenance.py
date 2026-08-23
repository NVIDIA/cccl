# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IR definition and cooperative storage-constructor analysis."""

from __future__ import annotations

import operator

from numba_cuda_mlir.numbair_transforms import ir

from cuda.coop._core import api as _portable_api

from .._temp_storage import TempStorage
from .._thread_data import ThreadData
from ._parameters import normalize_dtype_param
from ._rewrite_support import (
    _INFERENCE_EXCEPTIONS,
    _MIN_TEMP_STORAGE_ALIGNMENT,
    CoopSinglePhaseRewriteError,
    _normalize_alignment,
    _TempStorageCtorSpec,
    _ThreadDataSpec,
)


class _ProvenanceRewrite:
    """Recognize and validate foundation storage constructors."""

    def __init__(self, state):
        super().__init__(state)
        self._state = state
        self._func_ir = state.func_ir
        self._block: ir.Block | None = None
        self._block_defs: dict[str, object] = {}
        self._thread_data_assigns: dict[ir.Assign, _ThreadDataSpec] = {}
        self._temp_storage_assigns: dict[ir.Assign, _TempStorageCtorSpec] = {}
        self._thread_data_func_vars: set[str] = set()
        self._temp_storage_func_vars: set[str] = set()

    def _infer_constant(self, value):
        return self._func_ir.infer_constant(value)

    def _lookup_definition(self, value):
        if isinstance(value, ir.Var):
            if value.name in self._block_defs:
                return self._block_defs[value.name]
            try:
                return self._func_ir.get_definition(value)
            except KeyError:
                return None
        return value

    def _resolve_attribute_chain(self, value):
        attributes: list[str] = []
        current = self._lookup_definition(value)
        if current is None:
            return None
        while isinstance(current, ir.Expr) and current.op == "getattr":
            attributes.append(current.attr)
            current = self._lookup_definition(current.value)
            if current is None:
                return None
        if not isinstance(current, (ir.Global, ir.FreeVar, ir.Const)):
            return None
        attributes.reverse()
        return current.value, attributes

    def _resolve_python_value(self, value):
        chain = self._resolve_attribute_chain(value)
        if chain is None:
            return None
        root, attributes = chain
        resolved = root
        try:
            for attribute in attributes:
                resolved = getattr(resolved, attribute)
        except (AttributeError, ImportError):
            return None
        return resolved

    def _is_common_root_member(self, value, name: str) -> bool:
        member = getattr(_portable_api, name)
        return (
            self._resolve_python_value(value) is member
            and getattr(member, "__cuda_coop_backend_member__", None) == name
        )

    def _is_qualified_member(self, value, name: str) -> bool:
        member = {"TempStorage": TempStorage, "ThreadData": ThreadData}[name]
        return self._resolve_python_value(value) is member

    def _is_thread_data_ctor_call(self, call: ir.Expr) -> bool:
        return self._is_common_root_member(
            call.func, "ThreadData"
        ) or self._is_qualified_member(call.func, "ThreadData")

    def _is_temp_storage_ctor_call(self, call: ir.Expr) -> bool:
        return self._is_common_root_member(
            call.func, "TempStorage"
        ) or self._is_qualified_member(call.func, "TempStorage")

    @staticmethod
    def _validate_call_shape(call: ir.Expr, *, operation: str) -> None:
        if call.vararg is not None or call.varkwarg is not None:
            raise CoopSinglePhaseRewriteError(
                f"{operation} does not support *args or **kwargs in device code."
            )
        keyword_names = [name for name, _ in call.kws]
        duplicates = sorted(
            name for name in set(keyword_names) if keyword_names.count(name) > 1
        )
        if duplicates:
            raise CoopSinglePhaseRewriteError(
                f"{operation} got duplicate keyword(s): {', '.join(duplicates)}"
            )

    def _constant(self, value_ref, *, context: str):
        try:
            return self._infer_constant(value_ref)
        except _INFERENCE_EXCEPTIONS:
            resolved = self._resolve_python_value(value_ref)
            if resolved is not None:
                return resolved
        raise CoopSinglePhaseRewriteError(f"{context} must be a compile-time literal.")

    def _extract_thread_data_spec(self, call: ir.Expr) -> _ThreadDataSpec:
        self._validate_call_shape(call, operation="ThreadData")
        is_common_root = self._is_common_root_member(call.func, "ThreadData")
        allowed_keywords = {"items_per_thread", "dtype"}
        if not is_common_root:
            allowed_keywords.update({"alignas", "alignment"})
        keyword_names = [name for name, _ in call.kws]
        unexpected = sorted(set(keyword_names) - allowed_keywords)
        if unexpected:
            scope = "cuda.coop" if is_common_root else "cuda.coop.numba_mlir"
            raise CoopSinglePhaseRewriteError(
                f"{scope}.ThreadData got unexpected keyword(s): "
                + ", ".join(unexpected)
            )
        if len(call.args) > 2:
            raise CoopSinglePhaseRewriteError(
                "ThreadData accepts at most items_per_thread and dtype "
                "positional arguments."
            )

        keywords = dict(call.kws)
        if call.args and "items_per_thread" in keywords:
            raise CoopSinglePhaseRewriteError(
                "ThreadData received items_per_thread both positionally and by keyword."
            )
        if len(call.args) == 2 and "dtype" in keywords:
            raise CoopSinglePhaseRewriteError(
                "ThreadData received dtype both positionally and by keyword."
            )
        items_ref = call.args[0] if call.args else keywords.get("items_per_thread")
        if items_ref is None:
            raise CoopSinglePhaseRewriteError("ThreadData requires items_per_thread.")
        raw_items = self._constant(items_ref, context="ThreadData items_per_thread")
        if isinstance(raw_items, bool):
            raise CoopSinglePhaseRewriteError("items_per_thread must be an integer")
        try:
            items_per_thread = operator.index(raw_items)
        except TypeError as exc:
            raise CoopSinglePhaseRewriteError(
                "items_per_thread must be an integer"
            ) from exc
        if items_per_thread <= 0:
            raise CoopSinglePhaseRewriteError(
                "items_per_thread must be a positive integer"
            )

        dtype_ref = call.args[1] if len(call.args) == 2 else keywords.get("dtype")
        if dtype_ref is None:
            raise CoopSinglePhaseRewriteError(
                "ThreadData dtype must be specified until a cooperative "
                "primitive provides dtype inference."
            )
        raw_dtype = self._constant(dtype_ref, context="ThreadData dtype")
        try:
            dtype = normalize_dtype_param(raw_dtype)
        except (TypeError, ValueError, AttributeError) as exc:
            raise CoopSinglePhaseRewriteError(str(exc)) from exc

        raw_alignment = 8
        alignment_values = []
        for name in ("alignas", "alignment"):
            if name in keywords:
                alignment_values.append(
                    self._constant(keywords[name], context=f"ThreadData {name}")
                )
        if len(alignment_values) == 2 and alignment_values[0] != alignment_values[1]:
            raise CoopSinglePhaseRewriteError(
                "ThreadData alignas and alignment must match when both are set"
            )
        if alignment_values:
            raw_alignment = alignment_values[0]
        alignment = _normalize_alignment(
            raw_alignment,
            context="ThreadData alignment",
            minimum=_MIN_TEMP_STORAGE_ALIGNMENT,
            promote_to_minimum=False,
        )
        return _ThreadDataSpec(items_per_thread, dtype, alignment)

    def _extract_temp_storage_ctor_spec(self, call: ir.Expr) -> _TempStorageCtorSpec:
        self._validate_call_shape(call, operation="TempStorage")
        parameter_names = ("size_in_bytes", "alignment", "auto_sync", "sharing")
        if len(call.args) > len(parameter_names):
            raise CoopSinglePhaseRewriteError(
                "TempStorage accepts at most size_in_bytes, alignment, "
                "auto_sync, and sharing positional arguments."
            )
        keywords = dict(call.kws)
        unexpected = sorted(set(keywords) - set(parameter_names))
        if unexpected:
            raise CoopSinglePhaseRewriteError(
                "TempStorage got unexpected keyword(s): " + ", ".join(unexpected)
            )
        values = dict(zip(parameter_names, call.args))
        for name, value_ref in call.kws:
            if name in values:
                raise CoopSinglePhaseRewriteError(
                    f"TempStorage got multiple values for argument {name!r}"
                )
            values[name] = value_ref

        size_in_bytes = None
        if "size_in_bytes" in values:
            raw_size = self._constant(
                values["size_in_bytes"], context="TempStorage size_in_bytes"
            )
            if raw_size is not None and (
                not isinstance(raw_size, int) or isinstance(raw_size, bool)
            ):
                raise CoopSinglePhaseRewriteError(
                    "TempStorage size_in_bytes must be an integer or None."
                )
            size_in_bytes = raw_size
            if size_in_bytes is not None and size_in_bytes <= 0:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage size_in_bytes must be a positive integer."
                )

        alignment = _MIN_TEMP_STORAGE_ALIGNMENT
        if "alignment" in values:
            raw_alignment = self._constant(
                values["alignment"], context="TempStorage alignment"
            )
            if raw_alignment is not None:
                alignment = _normalize_alignment(
                    raw_alignment,
                    context="TempStorage alignment",
                    minimum=_MIN_TEMP_STORAGE_ALIGNMENT,
                )

        auto_sync = None
        if "auto_sync" in values:
            auto_sync = self._constant(
                values["auto_sync"], context="TempStorage auto_sync"
            )
            if auto_sync is not None and not isinstance(auto_sync, bool):
                raise CoopSinglePhaseRewriteError(
                    "TempStorage auto_sync must be None/True/False."
                )

        sharing = "shared"
        if "sharing" in values:
            sharing = self._constant(values["sharing"], context="TempStorage sharing")
            if not isinstance(sharing, str):
                raise CoopSinglePhaseRewriteError(
                    "TempStorage sharing must be a string: 'shared' or 'exclusive'."
                )
            sharing = sharing.strip().lower()
        if sharing not in {"shared", "exclusive"}:
            raise CoopSinglePhaseRewriteError(
                "TempStorage sharing must be 'shared' or 'exclusive'."
            )
        if sharing == "exclusive" and auto_sync is True:
            raise CoopSinglePhaseRewriteError(
                "TempStorage with sharing='exclusive' does not support auto_sync=True."
            )
        if sharing != "shared" or auto_sync is False:
            raise CoopSinglePhaseRewriteError(
                "TempStorage non-default sharing or auto_sync requires a "
                "cooperative primitive to consume the storage descriptor."
            )
        return _TempStorageCtorSpec(
            size_in_bytes=size_in_bytes,
            alignment=alignment,
        )


__all__ = ["_ProvenanceRewrite"]
