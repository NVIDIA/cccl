# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IR definition, payload-provenance, and constructor analysis.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from enum import Enum

from cuda.coop._core import StorageOwnership, SynchronizationScope

from .._temp_storage import TempStorage
from .._thread_data import ThreadData
from ._rewrite_support import (
    _INFERENCE_EXCEPTIONS,
    _MIN_TEMP_STORAGE_ALIGNMENT,
    _UNRESOLVED,
    CoopSinglePhaseRewriteError,
    _align_up,
    _cuda_module,
    _default_temp_storage_alignment,
    _dtype_values_match,
    _normalize_temp_storage_alignment,
    _phi_incoming_values,
    _portable_api,
    _ResolvedCallTarget,
    _RewriteMatch,
    _TempStorageCtorSpec,
    _TempStorageGlobalPlan,
    _TempStoragePlan,
    _TempStorageRequirementSummary,
    _TempStorageSlice,
    _ThreadDataSpec,
    _validate_temp_storage_alignment,
    factory_operation,
    ir,
    normalize_dtype_param,
    operator,
)
from ._scalar_provenance import (
    try_resolve_static_scalar,
    try_resolve_static_scalar_provenance,
)


class _ProvenanceRewrite:
    @staticmethod
    def _require_matching_items_per_thread(
        op_name: str,
        lhs_name: str,
        lhs_spec: _ThreadDataSpec | None,
        rhs_name: str,
        rhs_spec: _ThreadDataSpec | None,
    ) -> None:
        if lhs_spec is None or rhs_spec is None:
            return
        lhs_items = lhs_spec.items_per_thread
        rhs_items = rhs_spec.items_per_thread
        if lhs_items is not None and rhs_items is not None and (lhs_items != rhs_items):
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase '{op_name}' requires {lhs_name}/{rhs_name} arrays to have matching items_per_thread."
            )

    def __init__(self, state, *, allow_launch_dim_deferral: bool = True):
        super().__init__(state)
        self._state = state
        self._allow_launch_dim_deferral = allow_launch_dim_deferral
        self._func_ir = state.func_ir
        self._block: ir.Block | None = None
        self._block_defs: dict[str, object] = {}
        self._matches: dict[ir.Assign, _RewriteMatch] = {}
        self._temp_storage_assigns: set[ir.Assign] = set()
        self._temp_storage_func_vars: set[str] = set()
        self._temp_storage_ctor_specs: dict[str, _TempStorageCtorSpec] = {}
        self._temp_storage_ctor_order: dict[str, int] = {}
        self._temp_storage_ctor_roots: dict[str, str] = {}
        self._thread_data_func_vars: set[str] = set()
        self._typed_group_payload_func_vars: set[str] = set()
        self._thread_data_specs: dict[str, _ThreadDataSpec] = {}
        self._func_ir_identity: int | None = None
        self._func_temp_storage_requirements: dict[
            str, _TempStorageRequirementSummary
        ] = {}
        self._temp_storage_plans: dict[str, _TempStoragePlan] = {}
        self._temp_storage_global_plan: _TempStorageGlobalPlan | None = None
        self._implicit_temp_storage_requirements = _TempStorageRequirementSummary()
        self._implicit_temp_storage_plan: _TempStoragePlan | None = None
        self._temp_storage_backing_var: ir.Var | None = None
        self._temp_storage_backing_emitted = False
        self._arg_type_map = self._build_arg_type_map()
        self._invocable_cache: dict[
            tuple[str, tuple[tuple[str, str, str], ...]], object
        ] = {}
        self._prebundled_specializations: dict[
            tuple[str, tuple[tuple[str, str, str], ...]],
            tuple[object, int | None, int | tuple[int, ...] | None],
        ] = {}
        self._deferred_launch_dim_inference = False

    def _infer_constant(self, value):
        if isinstance(value, ir.Var):
            definition = self._block_defs.get(value.name)
            if isinstance(definition, (ir.Const, ir.Global, ir.FreeVar)):
                return definition.value
        return self._func_ir.infer_constant(value)

    def _resolve_static_scalar_value(
        self,
        value,
    ):
        """Resolve only scalar values with explicitly static IR provenance."""

        arg_types = tuple(getattr(self._state, "args", ()) or ())
        resolved, scalar = try_resolve_static_scalar(
            value,
            definitions=self._lookup_definitions,
            argument_type=lambda index: (
                arg_types[index] if 0 <= index < len(arg_types) else None
            ),
        )
        return scalar if resolved else _UNRESOLVED

    def _resolve_static_scalar_provenance(self, value):
        arg_types = tuple(getattr(self._state, "args", ()) or ())
        resolved, scalar = try_resolve_static_scalar_provenance(
            value,
            definitions=self._lookup_definitions,
            argument_type=lambda index: (
                arg_types[index] if 0 <= index < len(arg_types) else None
            ),
        )
        return scalar if resolved else _UNRESOLVED

    def _build_arg_type_map(self) -> dict[str, object]:
        arg_names = tuple(getattr(self._func_ir, "arg_names", ()) or ())
        arg_types = tuple(getattr(self._state, "args", ()) or ())
        if len(arg_names) != len(arg_types):
            return {}
        return dict(zip(arg_names, arg_types))

    def _lookup_definition(self, value):
        if isinstance(value, ir.Var):
            if value.name in self._block_defs:
                return self._block_defs[value.name]
            try:
                return self._func_ir.get_definition(value)
            except KeyError:
                return None
        if isinstance(value, str):
            if value in self._block_defs:
                return self._block_defs[value]
            try:
                return self._func_ir.get_definition(value)
            except KeyError:
                return None
        return value

    def _lookup_definitions(self, value) -> list[object]:
        defs: list[object] = []
        seen_ids: set[int] = set()

        def add(candidate) -> None:
            if candidate is None:
                return
            cid = id(candidate)
            if cid in seen_ids:
                return
            seen_ids.add(cid)
            defs.append(candidate)

        if isinstance(value, ir.Var):
            if value.name in self._block_defs:
                add(self._block_defs[value.name])
            for definition in (getattr(self._func_ir, "_definitions", {}) or {}).get(
                value.name, ()
            ):
                add(definition)
            return defs
        if isinstance(value, str):
            if value in self._block_defs:
                add(self._block_defs[value])
            for definition in (getattr(self._func_ir, "_definitions", {}) or {}).get(
                value, ()
            ):
                add(definition)
            return defs
        return [value]

    def _resolve_attribute_chain(self, func_var):
        attrs: list[str] = []
        current = self._lookup_definition(func_var)
        if current is None:
            return None
        while isinstance(current, ir.Expr) and current.op == "getattr":
            attrs.append(current.attr)
            current = self._lookup_definition(current.value)
            if current is None:
                return None
        if isinstance(current, (ir.Global, ir.FreeVar, ir.Const)):
            root = current.value
        else:
            return None
        attrs.reverse()
        return (root, attrs)

    def _resolve_python_value(self, value):
        chain = self._resolve_attribute_chain(value)
        if chain is None:
            return None
        root, attrs = chain
        obj = root
        try:
            for attr in attrs:
                obj = getattr(obj, attr)
        except (AttributeError, ImportError):
            return None
        return obj

    def _is_common_root_member(self, value, name: str) -> bool:
        member = getattr(_portable_api, name)
        return (
            self._resolve_python_value(value) is member
            and getattr(member, "__cuda_coop_backend_member__", None) == name
        )

    def _is_supported_factory(self, obj) -> bool:
        if not callable(obj):
            return False
        metadata = factory_operation(obj)
        if metadata is None:
            return False
        from ._operations import rewrite_operation

        spec = rewrite_operation(metadata.operation)
        if spec is None:
            return False
        return metadata.namespace in spec.factory_namespaces

    def _resolve_factory_from_var(self, func_var):
        direct = None
        direct_def = self._lookup_definition(func_var)
        if isinstance(direct_def, (ir.Global, ir.FreeVar, ir.Const)):
            direct = direct_def.value
        elif callable(direct_def):
            direct = direct_def
        elif direct_def is None:
            try:
                direct = self._infer_constant(func_var)
            except _INFERENCE_EXCEPTIONS:
                direct = None
        return direct if self._is_supported_factory(direct) else None

    def _extract_1d_extent_literal(self, value_ref):
        try:
            value = self._infer_constant(value_ref)
        except _INFERENCE_EXCEPTIONS:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, tuple) and len(value) == 1 and isinstance(value[0], int):
            return int(value[0])
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], int):
            return int(value[0])
        return None

    def _is_temp_storage_ctor_call(self, call: ir.Expr) -> bool:
        if self._is_common_root_member(call.func, "TempStorage"):
            return True
        return self._resolve_python_value(call.func) is TempStorage

    def _is_thread_data_ctor_call(self, call: ir.Expr) -> bool:
        if self._is_common_root_member(call.func, "ThreadData"):
            return True
        return self._resolve_python_value(call.func) is ThreadData

    def _is_typed_group_payload_ctor_call(self, call: ir.Expr) -> bool:
        chain = self._resolve_attribute_chain(call.func)
        if chain is None:
            return False
        root, attrs = chain
        if attrs:
            return False
        from ._group_planner_support import _typed_group_payload_like

        return root is _typed_group_payload_like

    def _is_typed_group_payload_var(self, value: ir.Var) -> bool:
        return any(
            (
                isinstance(definition, ir.Expr)
                and definition.op == "call"
                and self._is_typed_group_payload_ctor_call(definition)
                for definition in self._lookup_definitions(value)
            )
        )

    def _extract_typed_group_payload_spec(
        self, call: ir.Expr, *, seen: set[str] | None = None
    ) -> _ThreadDataSpec:
        if seen is None:
            seen = set()
        if len(call.args) not in {3, 4} or call.kws:
            raise CoopSinglePhaseRewriteError(
                "typed group payload marker requires prototype, array-kind, dtype-policy, and optional explicit-extent arguments"
            )
        prototype, is_array_ref, dtype_policy_ref = call.args[:3]
        if not isinstance(prototype, ir.Var):
            raise CoopSinglePhaseRewriteError(
                "typed group payload prototype must be a variable"
            )
        try:
            is_array = self._infer_constant(is_array_ref)
            dtype_policy = self._infer_constant(dtype_policy_ref)
        except _INFERENCE_EXCEPTIONS as exc:
            raise CoopSinglePhaseRewriteError(
                "typed group payload shape and dtype policy must be compile-time constants"
            ) from exc
        if not isinstance(is_array, bool):
            raise CoopSinglePhaseRewriteError(
                "typed group payload array-kind must be a compile-time bool"
            )
        from ._group_planner_support import _PAYLOAD_DTYPE_LIKE

        if dtype_policy != _PAYLOAD_DTYPE_LIKE:
            raise CoopSinglePhaseRewriteError(
                f"unknown typed group payload dtype policy {dtype_policy!r}"
            )
        prototype_spec = self._resolve_array_spec_from_var(prototype, seen=set(seen))
        if len(call.args) == 4:
            try:
                items_per_thread = self._infer_constant(call.args[3])
            except _INFERENCE_EXCEPTIONS as exc:
                raise CoopSinglePhaseRewriteError(
                    "typed group payload explicit extent must be a compile-time positive integer"
                ) from exc
            if (
                isinstance(items_per_thread, bool)
                or not isinstance(items_per_thread, int)
                or items_per_thread < 1
            ):
                raise CoopSinglePhaseRewriteError(
                    "typed group payload explicit extent must be a compile-time positive integer"
                )
        elif is_array:
            items_per_thread = (
                prototype_spec.items_per_thread if prototype_spec is not None else None
            )
        else:
            items_per_thread = 1
        dtype = prototype_spec.dtype if prototype_spec is not None else None
        if dtype is None:
            dtype = self._resolve_var_dtype(prototype)
        return _ThreadDataSpec(
            items_per_thread=items_per_thread,
            dtype=dtype,
            common_root=prototype_spec.common_root
            if prototype_spec is not None
            else False,
        )

    def _extract_thread_data_spec(self, call: ir.Expr) -> _ThreadDataSpec:
        kw_map = {name: value for name, value in call.kws}
        is_common_root = self._is_common_root_member(call.func, "ThreadData")
        allowed_keywords = {"items_per_thread", "dtype"}
        if not is_common_root:
            allowed_keywords.add("alignas")
        unexpected_keywords = sorted(set(kw_map) - allowed_keywords)
        if unexpected_keywords:
            names = ", ".join(unexpected_keywords)
            scope = "cuda.coop" if is_common_root else "cuda.coop.numba_mlir"
            raise CoopSinglePhaseRewriteError(
                f"{scope}.ThreadData got unexpected keyword(s): {names}"
            )
        extent_refs = []
        if call.args:
            extent_refs.append(("positional items_per_thread", call.args[0]))
        if "items_per_thread" in kw_map:
            extent_refs.append(("items_per_thread", kw_map["items_per_thread"]))
        if len(call.args) > 2:
            raise CoopSinglePhaseRewriteError(
                "coop.ThreadData accepts at most items_per_thread and dtype positional arguments."
            )
        if len(extent_refs) > 1:
            names = " and ".join((name for name, _ in extent_refs))
            raise CoopSinglePhaseRewriteError(
                f"coop.ThreadData received both {names}; specify only one."
            )
        if not extent_refs:
            raise CoopSinglePhaseRewriteError(
                "coop.ThreadData requires items_per_thread."
            )
        items_ref = extent_refs[0][1]
        dtype_ref = None
        if len(call.args) == 2:
            dtype_ref = call.args[1]
        if "dtype" in kw_map:
            if dtype_ref is not None:
                raise CoopSinglePhaseRewriteError(
                    "coop.ThreadData received dtype both positionally and by keyword."
                )
            dtype_ref = kw_map["dtype"]
        alignment = None
        alignment_ref = kw_map.get("alignas")
        if alignment_ref is not None:
            try:
                raw_alignment = self._infer_constant(alignment_ref)
            except _INFERENCE_EXCEPTIONS as exc:
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.ThreadData alignment must be a compile-time positive integer"
                ) from exc
            if isinstance(raw_alignment, bool):
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.ThreadData alignment must be a compile-time positive integer"
                )
            try:
                alignment = operator.index(raw_alignment)
            except TypeError as exc:
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.ThreadData alignment must be a compile-time positive integer"
                ) from exc
            if alignment < 1:
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.ThreadData alignment must be a compile-time positive integer"
                )
            if alignment & (alignment - 1):
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.ThreadData alignment must be a power of 2"
                )
            if alignment % _MIN_TEMP_STORAGE_ALIGNMENT:
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.ThreadData alignment must be a multiple "
                    f"of {_MIN_TEMP_STORAGE_ALIGNMENT}"
                )
        try:
            raw_items_per_thread = self._infer_constant(items_ref)
        except _INFERENCE_EXCEPTIONS as exc:
            raise CoopSinglePhaseRewriteError(
                "items_per_thread must be a compile-time integer"
            ) from exc
        if isinstance(raw_items_per_thread, bool):
            raise CoopSinglePhaseRewriteError("items_per_thread must be an integer")
        try:
            items_per_thread = operator.index(raw_items_per_thread)
        except TypeError as exc:
            raise CoopSinglePhaseRewriteError(
                "items_per_thread must be an integer"
            ) from exc
        if items_per_thread <= 0:
            raise CoopSinglePhaseRewriteError(
                "items_per_thread must be a positive integer"
            )
        dtype = None
        if dtype_ref is not None:
            dtype = self._resolve_dtype_ref(dtype_ref)
            if any((dtype is alias for alias in (bool, int, float, complex))):
                dtype = normalize_dtype_param(dtype)
        return _ThreadDataSpec(
            items_per_thread=items_per_thread,
            dtype=dtype,
            common_root=is_common_root,
            alignment=alignment,
        )

    @staticmethod
    def _merge_thread_data_specs(
        existing: _ThreadDataSpec | None, observed: _ThreadDataSpec
    ) -> _ThreadDataSpec:
        if existing is None:
            return observed
        if (
            existing.items_per_thread is not None
            and observed.items_per_thread is not None
            and (existing.items_per_thread != observed.items_per_thread)
        ):
            raise CoopSinglePhaseRewriteError(
                "Inconsistent items_per_thread across merged coop.ThreadData aliases."
            )
        if (
            existing.dtype is not None
            and observed.dtype is not None
            and (existing.dtype != observed.dtype)
        ):
            raise CoopSinglePhaseRewriteError(
                "Inconsistent dtype across merged coop.ThreadData aliases."
            )
        items_per_thread = existing.items_per_thread
        if items_per_thread is None:
            items_per_thread = observed.items_per_thread
        dtype = existing.dtype
        if dtype is None:
            dtype = observed.dtype
        alignment = existing.alignment
        if alignment is None:
            alignment = observed.alignment
        elif observed.alignment is not None and alignment != observed.alignment:
            alignment = None
        return _ThreadDataSpec(
            items_per_thread=items_per_thread,
            dtype=dtype,
            common_root=existing.common_root or observed.common_root,
            alignment=alignment,
        )

    @staticmethod
    def _merge_temp_storage_ctor_specs(
        existing: _TempStorageCtorSpec | None, observed: _TempStorageCtorSpec
    ) -> _TempStorageCtorSpec:
        if existing is None:
            return observed
        if existing != observed:
            raise CoopSinglePhaseRewriteError(
                "Inconsistent TempStorage constructor metadata across merged aliases."
            )
        return existing

    def _record_inferred_thread_data_dtype(
        self, value: ir.Var, dtype, seen: set[str] | None = None
    ) -> None:
        if not isinstance(value, ir.Var) or dtype is None:
            return
        if seen is None:
            seen = set()
        if value.name in seen:
            return
        seen.add(value.name)
        spec = self._thread_data_specs.get(value.name)
        if spec is None:
            spec = self._resolve_thread_data_spec(value)
            if spec is not None:
                self._thread_data_specs[value.name] = spec
        if spec is not None:
            if spec.dtype is None:
                self._thread_data_specs[value.name] = _ThreadDataSpec(
                    items_per_thread=spec.items_per_thread,
                    dtype=dtype,
                    common_root=spec.common_root,
                    alignment=spec.alignment,
                )
            elif spec.dtype != dtype:
                raise CoopSinglePhaseRewriteError(
                    "Inconsistent inferred dtype for coop.ThreadData usage."
                )
        for definition in self._lookup_definitions(value):
            if isinstance(definition, ir.Var):
                self._record_inferred_thread_data_dtype(definition, dtype, seen)
                continue
            if not isinstance(definition, ir.Expr):
                continue
            if definition.op == "cast":
                cast_value = getattr(definition, "value", None)
                if isinstance(cast_value, ir.Var):
                    self._record_inferred_thread_data_dtype(cast_value, dtype, seen)
            elif definition.op == "phi":
                for incoming in _phi_incoming_values(definition):
                    if isinstance(incoming, ir.Var):
                        self._record_inferred_thread_data_dtype(incoming, dtype, seen)

    def _extract_temp_storage_ctor_spec(self, call: ir.Expr) -> _TempStorageCtorSpec:
        kw_map = {name: value for name, value in call.kws}
        parameter_names = ("size_in_bytes", "alignment", "auto_sync", "sharing")
        if len(call.args) > len(parameter_names):
            raise CoopSinglePhaseRewriteError(
                "TempStorage accepts at most size_in_bytes, alignment, auto_sync, and sharing positional arguments."
            )
        unexpected_keywords = sorted(set(kw_map) - set(parameter_names))
        if unexpected_keywords:
            names = ", ".join(unexpected_keywords)
            raise CoopSinglePhaseRewriteError(
                f"TempStorage got unexpected keyword(s): {names}"
            )
        refs = dict(zip(parameter_names, call.args))
        for name, value_ref in call.kws:
            if name in refs:
                raise CoopSinglePhaseRewriteError(
                    f"TempStorage got multiple values for argument {name!r}"
                )
            refs[name] = value_ref
        size_ref = refs.get("size_in_bytes")
        alignment_ref = refs.get("alignment")
        auto_sync_ref = refs.get("auto_sync")
        sharing_ref = refs.get("sharing")

        def infer_constant(value_ref, *, name: str):
            try:
                return self._infer_constant(value_ref)
            except _INFERENCE_EXCEPTIONS as exc:
                raise CoopSinglePhaseRewriteError(
                    f"TempStorage {name} must be a compile-time literal."
                ) from exc

        size_in_bytes = None
        if size_ref is not None:
            raw_size_in_bytes = infer_constant(size_ref, name="size_in_bytes")
            if raw_size_in_bytes is not None and (
                not isinstance(raw_size_in_bytes, int)
                or isinstance(raw_size_in_bytes, bool)
            ):
                raise CoopSinglePhaseRewriteError(
                    "TempStorage size_in_bytes must be an integer or None."
                )
            size_in_bytes = raw_size_in_bytes
            if size_in_bytes is not None and size_in_bytes <= 0:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage size_in_bytes must be a positive integer."
                )
        alignment = None
        if alignment_ref is not None:
            raw_alignment = infer_constant(alignment_ref, name="alignment")
            if raw_alignment is not None and (
                not isinstance(raw_alignment, int) or isinstance(raw_alignment, bool)
            ):
                raise CoopSinglePhaseRewriteError(
                    "TempStorage alignment must be an integer or None."
                )
            if raw_alignment is not None:
                alignment = _normalize_temp_storage_alignment(raw_alignment)
        auto_sync = None
        if auto_sync_ref is not None:
            auto_sync = infer_constant(auto_sync_ref, name="auto_sync")
            if auto_sync is not None and (not isinstance(auto_sync, bool)):
                raise CoopSinglePhaseRewriteError(
                    "TempStorage auto_sync must be None/True/False."
                )
        sharing = "shared"
        if sharing_ref is not None:
            sharing = infer_constant(sharing_ref, name="sharing")
            if not isinstance(sharing, str) or isinstance(sharing, Enum):
                raise CoopSinglePhaseRewriteError(
                    "TempStorage sharing must be a string: 'shared' or 'exclusive'."
                )
            sharing = sharing.strip().lower()
        if sharing not in {"shared", "exclusive"}:
            raise CoopSinglePhaseRewriteError(
                "TempStorage sharing must be 'shared' or 'exclusive'."
            )
        return _TempStorageCtorSpec(
            size_in_bytes=size_in_bytes,
            alignment=alignment,
            auto_sync=auto_sync,
            sharing=sharing,
        )

    def _collect_temp_storage_ctor_keys(
        self, value: ir.Var, seen: set[str]
    ) -> set[str]:
        if not isinstance(value, ir.Var):
            return set()
        if value.name in seen:
            return set()
        if value.name in self._temp_storage_ctor_specs:
            return {value.name}
        seen.add(value.name)
        keys: set[str] = set()
        for definition in self._lookup_definitions(value):
            if isinstance(definition, ir.Expr):
                if definition.op == "call" and self._is_temp_storage_ctor_call(
                    definition
                ):
                    spec = self._extract_temp_storage_ctor_spec(definition)
                    self._temp_storage_ctor_specs[value.name] = (
                        self._merge_temp_storage_ctor_specs(
                            self._temp_storage_ctor_specs.get(value.name), spec
                        )
                    )
                    keys.add(value.name)
                    continue
                if definition.op == "cast":
                    cast_value = getattr(definition, "value", None)
                    if isinstance(cast_value, ir.Var):
                        keys.update(
                            self._collect_temp_storage_ctor_keys(cast_value, seen)
                        )
                    continue
                if definition.op == "phi":
                    for incoming in _phi_incoming_values(definition):
                        if isinstance(incoming, ir.Var):
                            keys.update(
                                self._collect_temp_storage_ctor_keys(incoming, seen)
                            )
                    continue
            if isinstance(definition, ir.Var):
                keys.update(self._collect_temp_storage_ctor_keys(definition, seen))
        return keys

    @staticmethod
    def _temp_storage_contract(
        spec: _TempStorageCtorSpec,
    ) -> tuple[int | None, int | None, bool, str]:
        auto_sync = (
            False
            if spec.sharing == "exclusive"
            else (True if spec.auto_sync is None else spec.auto_sync)
        )
        return (
            spec.size_in_bytes,
            spec.alignment,
            auto_sync,
            spec.sharing,
        )

    def _canonical_temp_storage_ctor_key(self, key: str) -> str:
        roots = getattr(self, "_temp_storage_ctor_roots", {})
        root = roots.get(key, key)
        while roots.get(root, root) != root:
            root = roots[root]
        if key != root:
            roots[key] = root
        return root

    def _merge_temp_storage_ctor_keys(self, keys: set[str]) -> str:
        roots = {self._canonical_temp_storage_ctor_key(key) for key in keys}
        contracts = {
            self._temp_storage_contract(self._temp_storage_ctor_specs[key])
            for key in roots
        }
        if len(contracts) != 1:
            ordered_keys = sorted(
                roots,
                key=lambda key: (
                    self._temp_storage_ctor_order.get(key, 1 << 30),
                    key,
                ),
            )
            names = ", ".join(ordered_keys)
            raise CoopSinglePhaseRewriteError(
                "TempStorage aliases have inconsistent contracts across "
                f"constructor instances ({names})."
            )
        canonical = min(
            roots,
            key=lambda key: (
                self._temp_storage_ctor_order.get(key, 1 << 30),
                key,
            ),
        )
        for key, root in tuple(self._temp_storage_ctor_roots.items()):
            if self._canonical_temp_storage_ctor_key(root) in roots:
                self._temp_storage_ctor_roots[key] = canonical
        for key in roots | keys:
            self._temp_storage_ctor_roots[key] = canonical
        return canonical

    def _resolve_temp_storage_ctor_key(self, value: ir.Var) -> str | None:
        if not isinstance(value, ir.Var):
            return None
        keys = self._collect_temp_storage_ctor_keys(value, seen=set())
        if not keys:
            return None
        return self._merge_temp_storage_ctor_keys(keys)

    def _resolve_temp_storage_plan(self, value: ir.Var) -> _TempStoragePlan | None:
        key = self._resolve_temp_storage_ctor_key(value)
        if key is None:
            return None
        if self._temp_storage_global_plan is None and self._temp_storage_ctor_specs:
            self._ensure_temp_storage_global_plan()
        return self._finalize_temp_storage_plan_for_var(key)

    @staticmethod
    def _temp_storage_domain_key(
        entry,
    ) -> tuple[object, ...]:
        lowering_plan = entry.lowering_plan
        if lowering_plan is None:
            return ("legacy-provider",)
        if lowering_plan.unsupported is not None:
            raise CoopSinglePhaseRewriteError(
                "cooperative provider storage received an unsupported "
                "group lowering plan."
            )
        topology = lowering_plan.topology
        synchronization = lowering_plan.synchronization
        storage = lowering_plan.temp_storage
        if topology is None or synchronization is None or storage is None:
            raise CoopSinglePhaseRewriteError(
                "cooperative provider storage requires complete group "
                "topology, synchronization, and storage contracts."
            )
        if storage.ownership is StorageOwnership.NONE:
            raise CoopSinglePhaseRewriteError(
                "a storage-bearing cooperative provider received a "
                "storage-free lowering plan."
            )
        if storage.instances != topology.instances or (
            storage.instance_index != topology.instance_index
        ):
            raise CoopSinglePhaseRewriteError(
                "cooperative provider storage layout disagrees with its group topology."
            )
        if storage.ownership is StorageOwnership.CALLER:
            # Explicit single-block storage keeps the established caller-owned
            # reuse layout. Topology domains apply only to implementation-owned
            # storage, where the backend controls every slice.
            return ("caller-storage",)
        domain = (
            topology.group_kind,
            topology.execution_scope.value,
            topology.logical_width,
            topology.instances,
            topology.instance_index,
            topology.thread_rank,
            synchronization.storage_reuse_barrier.value,
        )
        if (
            synchronization.storage_reuse_barrier is SynchronizationScope.NONE
            and topology.execution_scope is not SynchronizationScope.NONE
        ):
            return (*domain, "non-reusable", entry.order)
        return domain

    def _layout_temp_storage_uses(
        self,
        uses,
        *,
        sharing: str,
    ) -> tuple[int, int, dict[int, _TempStorageSlice]]:
        ordered_uses = sorted(uses, key=lambda entry: entry.order)
        required_alignment = max(
            _MIN_TEMP_STORAGE_ALIGNMENT,
            *(max(1, int(entry.alignment)) for entry in ordered_uses),
        )
        domains: dict[tuple[object, ...], list[object]] = {}
        for entry in ordered_uses:
            domain_key = (
                ("exclusive", entry.order)
                if sharing == "exclusive"
                else self._temp_storage_domain_key(entry)
            )
            domains.setdefault(domain_key, []).append(entry)

        required_size = 0
        slices_by_call_id: dict[int, _TempStorageSlice] = {}
        for domain_uses in domains.values():
            domain_alignment = max(
                _MIN_TEMP_STORAGE_ALIGNMENT,
                *(max(1, int(entry.alignment)) for entry in domain_uses),
            )
            per_instance_size = max(int(entry.size_in_bytes) for entry in domain_uses)
            instance_stride = _align_up(per_instance_size, domain_alignment)
            first_plan = domain_uses[0].lowering_plan
            instances = (
                1
                if first_plan is None or first_plan.topology is None
                else first_plan.topology.instances
            )
            required_size = _align_up(required_size, domain_alignment)
            domain_offset = required_size
            for entry in domain_uses:
                slices_by_call_id[id(entry.call_assign)] = _TempStorageSlice(
                    offset=domain_offset,
                    size_in_bytes=int(entry.size_in_bytes),
                    stride=instance_stride,
                    instances=instances,
                    lowering_plan=entry.lowering_plan,
                )
            required_size += (
                per_instance_size if instances == 1 else instance_stride * instances
            )
        return required_size, required_alignment, slices_by_call_id

    def _finalize_temp_storage_plan_for_var(self, var_name: str) -> _TempStoragePlan:
        var_name = self._canonical_temp_storage_ctor_key(var_name)
        cached = self._temp_storage_plans.get(var_name)
        if cached is not None:
            return cached
        ctor_spec = self._temp_storage_ctor_specs.get(var_name)
        if ctor_spec is None:
            raise CoopSinglePhaseRewriteError(
                f"Missing TempStorage constructor metadata for variable '{var_name}'."
            )
        requirements = self._func_temp_storage_requirements.get(var_name)
        uses = list(requirements.uses) if requirements is not None else []
        uses.sort(key=lambda entry: entry.order)
        if uses:
            (
                required_size,
                required_alignment,
                slices_by_call_id,
            ) = self._layout_temp_storage_uses(
                uses,
                sharing=ctor_spec.sharing,
            )
        else:
            required_size = 0
            required_alignment = max(
                _MIN_TEMP_STORAGE_ALIGNMENT, int(ctor_spec.alignment or 1)
            )
            slices_by_call_id = {}
        if ctor_spec.size_in_bytes is None:
            if required_size <= 0:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage size_in_bytes must be specified until a "
                    "cooperative primitive provides a storage requirement."
                )
            size_in_bytes = required_size
        else:
            size_in_bytes = int(ctor_spec.size_in_bytes)
        if size_in_bytes <= 0:
            raise CoopSinglePhaseRewriteError(
                "TempStorage size_in_bytes must be a positive integer."
            )
        if required_size > 0 and size_in_bytes < required_size:
            raise CoopSinglePhaseRewriteError(
                f"TempStorage size_in_bytes is smaller than required by primitive uses ({size_in_bytes} < {required_size})."
            )
        if ctor_spec.alignment is None:
            alignment = _default_temp_storage_alignment(required_alignment)
        else:
            alignment = int(ctor_spec.alignment)
        _validate_temp_storage_alignment(alignment)
        if required_alignment > 0 and alignment < required_alignment:
            raise CoopSinglePhaseRewriteError(
                f"TempStorage alignment is smaller than required by primitive uses ({alignment} < {required_alignment})."
            )
        if ctor_spec.sharing == "exclusive":
            if ctor_spec.auto_sync is True:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage with sharing='exclusive' does not support auto_sync=True."
                )
            auto_sync = False
        else:
            auto_sync = True if ctor_spec.auto_sync is None else ctor_spec.auto_sync
        if not uses and (ctor_spec.sharing != "shared" or ctor_spec.auto_sync is False):
            raise CoopSinglePhaseRewriteError(
                "TempStorage non-default sharing or auto_sync requires a "
                "cooperative primitive to consume the storage descriptor."
            )
        plan = _TempStoragePlan(
            size_in_bytes=size_in_bytes,
            alignment=alignment,
            sharing=ctor_spec.sharing,
            auto_sync=auto_sync,
            slices_by_call_id=slices_by_call_id,
        )
        self._temp_storage_plans[var_name] = plan
        return plan

    def _is_local_array_ctor_call(self, call: ir.Expr) -> bool:
        return self._resolve_python_value(call.func) is _cuda_module.local.array

    def _extract_local_array_spec(self, call: ir.Expr) -> _ThreadDataSpec:
        kw_map = {name: value for name, value in call.kws}
        items_ref = None
        if call.args:
            items_ref = call.args[0]
        elif "shape" in kw_map:
            items_ref = kw_map["shape"]
        dtype_ref = None
        if len(call.args) >= 2:
            dtype_ref = call.args[1]
        elif "dtype" in kw_map:
            dtype_ref = kw_map["dtype"]
        items_per_thread = None
        if items_ref is not None:
            items_per_thread = self._extract_1d_extent_literal(items_ref)
        dtype = None
        if dtype_ref is not None:
            dtype = self._resolve_dtype_ref(dtype_ref)
        return _ThreadDataSpec(items_per_thread=items_per_thread, dtype=dtype)

    def _is_shared_array_ctor_call(self, call: ir.Expr) -> bool:
        return self._resolve_python_value(call.func) is _cuda_module.shared.array

    def _extract_shared_array_spec(self, call: ir.Expr) -> _ThreadDataSpec:
        kw_map = {name: value for name, value in call.kws}
        shape_ref = None
        if call.args:
            shape_ref = call.args[0]
        elif "shape" in kw_map:
            shape_ref = kw_map["shape"]
        dtype_ref = None
        if len(call.args) >= 2:
            dtype_ref = call.args[1]
        elif "dtype" in kw_map:
            dtype_ref = kw_map["dtype"]
        extent = None
        if shape_ref is not None:
            extent = self._extract_1d_extent_literal(shape_ref)
        dtype = None
        if dtype_ref is not None:
            dtype = self._resolve_dtype_ref(dtype_ref)
        return _ThreadDataSpec(items_per_thread=extent, dtype=dtype)

    def _resolve_array_spec_from_var(
        self, value: ir.Var, seen: set[str]
    ) -> _ThreadDataSpec | None:
        if not isinstance(value, ir.Var):
            return None
        if value.name in seen:
            return None
        seen.add(value.name)
        merged: _ThreadDataSpec | None = None
        for definition in self._lookup_definitions(value):
            candidate: _ThreadDataSpec | None = None
            if isinstance(definition, ir.Expr):
                if definition.op == "call":
                    if self._is_thread_data_ctor_call(definition):
                        candidate = self._extract_thread_data_spec(definition)
                    elif self._is_typed_group_payload_ctor_call(definition):
                        candidate = self._extract_typed_group_payload_spec(
                            definition, seen=seen
                        )
                    elif self._is_local_array_ctor_call(definition):
                        candidate = self._extract_local_array_spec(definition)
                    elif self._is_shared_array_ctor_call(definition):
                        candidate = self._extract_shared_array_spec(definition)
                elif definition.op == "cast":
                    cast_value = getattr(definition, "value", None)
                    if isinstance(cast_value, ir.Var):
                        candidate = self._resolve_array_spec_from_var(cast_value, seen)
                elif definition.op == "static_getitem":
                    for item in self._resolve_static_tuple_item_vars(definition):
                        item_spec = self._resolve_array_spec_from_var(item, set(seen))
                        if item_spec is None:
                            continue
                        merged = self._merge_thread_data_specs(merged, item_spec)
                    continue
                elif definition.op == "phi":
                    for incoming in _phi_incoming_values(definition):
                        if not isinstance(incoming, ir.Var):
                            continue
                        incoming_spec = self._resolve_array_spec_from_var(
                            incoming, set(seen)
                        )
                        if incoming_spec is None:
                            continue
                        merged = self._merge_thread_data_specs(merged, incoming_spec)
                    continue
            elif isinstance(definition, ir.Var):
                candidate = self._resolve_array_spec_from_var(definition, seen)
            if candidate is not None:
                merged = self._merge_thread_data_specs(merged, candidate)
        return merged

    def _resolve_thread_data_spec_from_var(
        self, value: ir.Var, seen: set[str]
    ) -> _ThreadDataSpec | None:
        if not isinstance(value, ir.Var):
            return None
        cached = self._thread_data_specs.get(value.name)
        if (
            cached is not None
            and cached.items_per_thread is not None
            and (cached.dtype is not None)
        ):
            return cached
        if value.name in seen:
            return None
        seen.add(value.name)
        merged: _ThreadDataSpec | None = cached
        for definition in self._lookup_definitions(value):
            candidate: _ThreadDataSpec | None = None
            if isinstance(definition, ir.Expr):
                if definition.op == "call":
                    if self._is_thread_data_ctor_call(definition):
                        candidate = self._extract_thread_data_spec(definition)
                    elif self._is_typed_group_payload_ctor_call(definition):
                        candidate = self._extract_typed_group_payload_spec(
                            definition, seen=seen
                        )
                    elif self._is_local_array_ctor_call(definition):
                        candidate = self._extract_local_array_spec(definition)
                elif definition.op == "cast":
                    cast_value = getattr(definition, "value", None)
                    if isinstance(cast_value, ir.Var):
                        candidate = self._resolve_thread_data_spec_from_var(
                            cast_value, seen
                        )
                elif definition.op == "static_getitem":
                    for item in self._resolve_static_tuple_item_vars(definition):
                        item_spec = self._resolve_thread_data_spec_from_var(
                            item, set(seen)
                        )
                        if item_spec is None:
                            continue
                        merged = self._merge_thread_data_specs(merged, item_spec)
                    continue
                elif definition.op == "phi":
                    for incoming in _phi_incoming_values(definition):
                        if not isinstance(incoming, ir.Var):
                            continue
                        incoming_spec = self._resolve_thread_data_spec_from_var(
                            incoming, set(seen)
                        )
                        if incoming_spec is None:
                            continue
                        merged = self._merge_thread_data_specs(merged, incoming_spec)
                    continue
            elif isinstance(definition, ir.Var):
                candidate = self._resolve_thread_data_spec_from_var(definition, seen)
            if candidate is not None:
                merged = self._merge_thread_data_specs(merged, candidate)
        if merged is not None:
            self._thread_data_specs[value.name] = merged
        return merged

    def _resolve_static_tuple_item_vars(self, definition: ir.Expr) -> list[ir.Var]:
        index = getattr(definition, "index", None)
        tuple_value = getattr(definition, "value", None)
        if not isinstance(index, int) or not isinstance(tuple_value, ir.Var):
            return []
        return self._resolve_tuple_item_vars(tuple_value, index, seen=set())

    def _resolve_tuple_item_vars(
        self, tuple_value: ir.Var, index: int, seen: set[str]
    ) -> list[ir.Var]:
        if tuple_value.name in seen:
            return []
        seen.add(tuple_value.name)
        items: list[ir.Var] = []
        for tuple_definition in self._lookup_definitions(tuple_value):
            if isinstance(tuple_definition, ir.Var):
                items.extend(
                    self._resolve_tuple_item_vars(
                        tuple_definition, index, seen=set(seen)
                    )
                )
                continue
            if not isinstance(tuple_definition, ir.Expr):
                continue
            if tuple_definition.op == "build_tuple":
                tuple_items = tuple(getattr(tuple_definition, "items", ()))
                if -len(tuple_items) <= index < len(tuple_items):
                    item = tuple_items[index]
                    if isinstance(item, ir.Var):
                        items.append(item)
                continue
            if tuple_definition.op in {"cast", "exhaust_iter"}:
                source = getattr(tuple_definition, "value", None)
                if isinstance(source, ir.Var):
                    items.extend(
                        self._resolve_tuple_item_vars(source, index, seen=set(seen))
                    )
                continue
            if tuple_definition.op == "phi":
                for incoming in _phi_incoming_values(tuple_definition):
                    if isinstance(incoming, ir.Var):
                        items.extend(
                            self._resolve_tuple_item_vars(
                                incoming, index, seen=set(seen)
                            )
                        )
        return items

    def _resolve_thread_data_spec(self, value: ir.Var) -> _ThreadDataSpec | None:
        if not isinstance(value, ir.Var):
            return None
        return self._resolve_thread_data_spec_from_var(value, seen=set())

    def _resolve_var_numba_type(self, value: ir.Var):
        typemap = getattr(self._state, "typemap", None)
        if isinstance(typemap, dict):
            mapped = typemap.get(value.name)
            if mapped is not None:
                return mapped
        if value.name in self._arg_type_map:
            return self._arg_type_map[value.name]
        definition = self._lookup_definition(value)
        if isinstance(definition, ir.Arg):
            arg_types = tuple(getattr(self._state, "args", ()) or ())
            if 0 <= definition.index < len(arg_types):
                return arg_types[definition.index]
        return None

    def _resolve_call_result_dtype(self, definition: ir.Expr):
        func_obj = None
        func_ref = getattr(definition, "func", None)
        if isinstance(func_ref, ir.Var):
            try:
                func_obj = self._infer_constant(func_ref)
            except (*_INFERENCE_EXCEPTIONS, ImportError):
                func_obj = None
            if func_obj is None:
                func_def = self._lookup_definition(func_ref)
                if isinstance(func_def, (ir.Global, ir.FreeVar, ir.Const)):
                    func_obj = func_def.value
        elif isinstance(func_ref, (ir.Global, ir.FreeVar, ir.Const)):
            func_obj = func_ref.value
        if func_obj is None:
            try:
                func_obj = self._resolve_python_value(func_ref)
            except _INFERENCE_EXCEPTIONS:
                func_obj = None
        if func_obj is None:
            return None
        from ._parameters import (
            _scalar_cast_dtype,
            _scalar_operator_result_dtype,
        )

        cast_dtype = _scalar_cast_dtype(func_obj)
        if cast_dtype is None:
            return None
        if len(definition.args) == 1 and isinstance(definition.args[0], ir.Var):
            inferred = _scalar_operator_result_dtype(
                func_obj,
                self._resolve_var_dtype(definition.args[0]),
            )
            if inferred is not None:
                return inferred
        return cast_dtype

    @staticmethod
    def _merge_scalar_dtypes(dtypes):
        candidates = list(dtypes)
        if not candidates or any(dtype is None for dtype in candidates):
            return None
        resolved = []
        for dtype in candidates:
            try:
                dtype = normalize_dtype_param(dtype)
            except (TypeError, ValueError):
                pass
            if any(_dtype_values_match(dtype, existing) for existing in resolved):
                continue
            resolved.append(dtype)
        return resolved[0] if len(resolved) == 1 else None

    def _cuda_index_dtype(self, definition: ir.Expr):
        if definition.op != "getattr":
            return None
        chain = self._resolve_attribute_chain(definition.value)
        if chain is None:
            return None
        root, attributes = chain
        full_attributes = (*attributes, definition.attr)
        if root is _cuda_module and full_attributes in {
            (index, component)
            for index in ("blockDim", "blockIdx", "gridDim", "threadIdx")
            for component in ("x", "y", "z")
        }:
            from numba_cuda_mlir import types as numba_mlir_types

            return numba_mlir_types.int32
        return None

    def _resolve_definition_dtype(self, definition, *, seen: set[str]):
        from ._parameters import (
            _python_scalar_dtype,
            _scalar_operator_result_dtype,
        )

        if isinstance(definition, ir.Arg):
            arg_types = tuple(getattr(self._state, "args", ()) or ())
            if 0 <= definition.index < len(arg_types):
                arg_type = arg_types[definition.index]
                return getattr(arg_type, "dtype", arg_type)
            return None
        if isinstance(definition, (ir.Const, ir.Global, ir.FreeVar)):
            return _python_scalar_dtype(definition.value)
        if isinstance(definition, ir.Var):
            return self._resolve_var_dtype(definition, seen=set(seen))
        if not isinstance(definition, ir.Expr):
            return None
        if definition.op in {"cast", "exhaust_iter"}:
            source = getattr(definition, "value", None)
            if isinstance(source, ir.Var):
                return self._resolve_var_dtype(source, seen=set(seen))
            return None
        if definition.op == "getattr":
            cuda_dtype = self._cuda_index_dtype(definition)
            if cuda_dtype is not None:
                return cuda_dtype
            if definition.attr == "dtype" and isinstance(definition.value, ir.Var):
                return self._resolve_var_dtype(definition.value, seen=set(seen))
            return None
        if definition.op in {"getitem", "static_getitem"}:
            base_value = getattr(definition, "value", None)
            if isinstance(base_value, ir.Var):
                return self._resolve_var_dtype(base_value, seen=set(seen))
            return None
        if definition.op in {"binop", "inplace_binop"}:
            lhs = getattr(definition, "lhs", None)
            rhs = getattr(definition, "rhs", None)
            lhs_dtype = (
                self._resolve_var_dtype(lhs, seen=set(seen))
                if isinstance(lhs, ir.Var)
                else None
            )
            rhs_dtype = (
                self._resolve_var_dtype(rhs, seen=set(seen))
                if isinstance(rhs, ir.Var)
                else None
            )
            return _scalar_operator_result_dtype(
                getattr(definition, "fn", None),
                lhs_dtype,
                rhs_dtype,
            )
        if definition.op == "unary":
            unary_value = getattr(definition, "value", None)
            if isinstance(unary_value, ir.Var):
                return _scalar_operator_result_dtype(
                    getattr(definition, "fn", None),
                    self._resolve_var_dtype(unary_value, seen=set(seen)),
                )
            return None
        if definition.op == "phi":
            return self._merge_scalar_dtypes(
                self._resolve_var_dtype(incoming, seen=set(seen))
                for incoming in _phi_incoming_values(definition)
                if isinstance(incoming, ir.Var)
            )
        if definition.op == "call":
            return self._resolve_call_result_dtype(definition)
        return None

    def _infer_thread_data_dtype_from_writes(self, value: ir.Var):
        spec = self._resolve_thread_data_spec(value)
        if spec is None:
            return None
        alias_names = {value.name}
        changed = True
        while changed:
            changed = False
            for block in self._func_ir.blocks.values():
                for stmt in block.body:
                    if not isinstance(stmt, ir.Assign):
                        continue
                    definition = stmt.value
                    sources: tuple[ir.Var, ...] = ()
                    if isinstance(definition, ir.Var):
                        sources = (definition,)
                    elif isinstance(definition, ir.Expr) and definition.op == "cast":
                        if isinstance(definition.value, ir.Var):
                            sources = (definition.value,)
                    elif isinstance(definition, ir.Expr) and definition.op == "phi":
                        sources = tuple(
                            (
                                incoming
                                for incoming in _phi_incoming_values(definition)
                                if isinstance(incoming, ir.Var)
                            )
                        )
                    source_names = {source.name for source in sources}
                    if stmt.target.name in alias_names or source_names & alias_names:
                        additions = {stmt.target.name, *source_names} - alias_names
                        if additions:
                            alias_names.update(additions)
                            changed = True
        inferred = None
        static_setitem_cls = getattr(ir, "StaticSetItem", None)
        for block in self._func_ir.blocks.values():
            for stmt in block.body:
                if isinstance(stmt, ir.SetItem) or (
                    static_setitem_cls is not None
                    and isinstance(stmt, static_setitem_cls)
                ):
                    target = getattr(stmt, "target", None)
                    rhs = getattr(stmt, "value", None)
                else:
                    continue
                if not isinstance(target, ir.Var) or target.name not in alias_names:
                    continue
                if not isinstance(rhs, ir.Var):
                    continue
                rhs_dtype = self._resolve_var_dtype(rhs)
                if rhs_dtype is None:
                    continue
                if inferred is None:
                    inferred = rhs_dtype
                    continue
                if inferred != rhs_dtype:
                    raise CoopSinglePhaseRewriteError(
                        "Failed to infer a consistent dtype from coop.ThreadData writes."
                    )
        if inferred is not None:
            self._record_inferred_thread_data_dtype(value, inferred)
        return inferred

    def _collect_thread_data_write_roots(
        self, value: ir.Var, seen: set[str] | None = None
    ) -> dict[str, ir.Var]:
        """Find concrete ThreadData constructors behind group payload markers."""
        if not isinstance(value, ir.Var):
            return {}
        if seen is None:
            seen = set()
        if value.name in seen:
            return {}
        seen.add(value.name)
        roots: dict[str, ir.Var] = {}
        for definition in self._lookup_definitions(value):
            sources: tuple[ir.Var, ...] = ()
            if isinstance(definition, ir.Var):
                sources = (definition,)
            elif isinstance(definition, ir.Expr):
                if definition.op == "call":
                    if self._is_thread_data_ctor_call(definition):
                        roots[value.name] = value
                        continue
                    if (
                        self._is_typed_group_payload_ctor_call(definition)
                        and definition.args
                    ):
                        prototype = definition.args[0]
                        if isinstance(prototype, ir.Var):
                            sources = (prototype,)
                elif definition.op in {"cast", "exhaust_iter"}:
                    source = getattr(definition, "value", None)
                    if isinstance(source, ir.Var):
                        sources = (source,)
                elif definition.op == "phi":
                    sources = tuple(
                        (
                            incoming
                            for incoming in _phi_incoming_values(definition)
                            if isinstance(incoming, ir.Var)
                        )
                    )
                elif definition.op == "static_getitem":
                    sources = tuple(self._resolve_static_tuple_item_vars(definition))
            for source in sources:
                roots.update(
                    self._collect_thread_data_write_roots(source, seen=set(seen))
                )
        return roots

    def _resolve_var_dtype(self, value: ir.Var, seen: set[str] | None = None):
        if seen is None:
            seen = set()
        if value.name in seen:
            return None
        seen.add(value.name)
        spec = self._resolve_thread_data_spec(value)
        if spec is not None and spec.dtype is not None:
            return spec.dtype
        var_type = self._resolve_var_numba_type(value)
        dtype = getattr(var_type, "dtype", None)
        if dtype is not None:
            return dtype
        if var_type is not None and hasattr(var_type, "bitwidth"):
            return var_type
        return self._merge_scalar_dtypes(
            self._resolve_definition_dtype(definition, seen=set(seen))
            for definition in self._lookup_definitions(value)
        )

    def _resolve_dtype_ref(self, value_ref):
        try:
            return self._infer_constant(value_ref)
        except _INFERENCE_EXCEPTIONS:
            pass
        if isinstance(value_ref, ir.Var):
            definition = self._lookup_definition(value_ref)
            if isinstance(definition, ir.Expr) and definition.op == "getattr":
                if definition.attr == "dtype" and isinstance(definition.value, ir.Var):
                    return self._resolve_var_dtype(definition.value)
            return self._resolve_var_dtype(value_ref)
        return None

    def _resolve_factory_kwarg_value(self, op_name: str, name: str, value_ref):
        from ._operations import rewrite_operation

        spec = rewrite_operation(op_name)
        if spec is not None and name in spec.dtype_factory_kwargs:
            dtype = self._resolve_dtype_ref(value_ref)
            if dtype is not None:
                return dtype
        try:
            return self._infer_constant(value_ref)
        except _INFERENCE_EXCEPTIONS:
            pass
        if isinstance(value_ref, ir.Var):
            from numba_cuda_mlir import types as numba_mlir_types

            value_type = self._arg_type_map.get(value_ref.name)
            definition = self._lookup_definition(value_ref)
            if isinstance(definition, ir.Arg):
                value_type = self._state.args[definition.index]
            if isinstance(value_type, numba_mlir_types.NoneType) or (
                isinstance(value_type, numba_mlir_types.Omitted)
                and value_type.value is None
            ):
                return None
        return _UNRESOLVED

    def _resolve_call_target(self, call: ir.Expr):
        factory = self._resolve_factory_from_var(call.func)
        if factory is not None:
            metadata = factory_operation(factory)
            assert metadata is not None
            return _ResolvedCallTarget(
                factory=factory,
                factory_metadata=metadata,
                func_var_name=call.func.name,
                func_var_name_extra=None,
                getitem_temp_storage=None,
            )
        func_def = self._lookup_definition(call.func)
        if not (
            isinstance(func_def, ir.Expr)
            and func_def.op in {"getitem", "static_getitem"}
        ):
            return None
        factory = self._resolve_factory_from_var(func_def.value)
        if factory is None:
            return None
        metadata = factory_operation(factory)
        assert metadata is not None
        getitem_temp_storage = getattr(func_def, "index", None)
        if not isinstance(getitem_temp_storage, ir.Var):
            getitem_temp_storage = getattr(func_def, "index_var", None)
        if not isinstance(getitem_temp_storage, ir.Var):
            raise CoopSinglePhaseRewriteError(
                f"coop single-phase getitem syntax expects a runtime temp-storage variable: '{factory.__name__}[temp_storage](...)'."
            )
        return _ResolvedCallTarget(
            factory=factory,
            factory_metadata=metadata,
            func_var_name=call.func.name,
            func_var_name_extra=func_def.value.name,
            getitem_temp_storage=getitem_temp_storage,
        )


__all__ = ["_ProvenanceRewrite"]
