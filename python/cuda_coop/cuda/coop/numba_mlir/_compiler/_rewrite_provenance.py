# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IR definition, payload-provenance, and constructor analysis.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

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
    _numba_typeof,
    _numba_types,
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
    np,
    operator,
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
        self._thread_data_func_vars: set[str] = set()
        self._typed_group_payload_func_vars: set[str] = set()
        self._thread_data_specs: dict[str, _ThreadDataSpec] = {}
        self._func_ir_identity: int | None = None
        self._func_temp_storage_requirements: dict[
            str, _TempStorageRequirementSummary
        ] = {}
        self._temp_storage_plans: dict[str, _TempStoragePlan] = {}
        self._temp_storage_global_plan: _TempStorageGlobalPlan | None = None
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
        spec = self._OP_SPECS.get(metadata.operation)
        if spec is None:
            return False
        expected_ns = spec["namespace"]
        return metadata.namespace == expected_ns

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
        from ._group_planner_support import (
            _PAYLOAD_DTYPE_INT32,
            _PAYLOAD_DTYPE_LIKE,
        )

        if dtype_policy not in {_PAYLOAD_DTYPE_INT32, _PAYLOAD_DTYPE_LIKE}:
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
        if dtype_policy == _PAYLOAD_DTYPE_INT32:
            dtype = _numba_types.int32
        else:
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
            allowed_keywords.update(("alignas", "alignment"))
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
        alignment_values = []
        for alignment_name in ("alignas", "alignment"):
            alignment_ref = kw_map.get(alignment_name)
            if alignment_ref is None:
                continue
            try:
                raw_alignment = self._infer_constant(alignment_ref)
            except _INFERENCE_EXCEPTIONS as exc:
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir.ThreadData alignment must be a compile-time positive integer"
                ) from exc
            if alignment_name == "alignment" and raw_alignment is None:
                continue
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
            alignment_values.append((alignment_name, alignment))

        if len(alignment_values) == 2 and (
            alignment_values[0][1] != alignment_values[1][1]
        ):
            raise CoopSinglePhaseRewriteError(
                "cuda.coop.numba_mlir.ThreadData alignas and alignment must match when both are set"
            )
        alignment = None
        if alignment_values:
            alignment = alignment_values[-1][1]
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
            if not isinstance(sharing, str):
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

    def _resolve_temp_storage_ctor_key(self, value: ir.Var) -> str | None:
        if not isinstance(value, ir.Var):
            return None
        keys = self._collect_temp_storage_ctor_keys(value, seen=set())
        if not keys:
            return None
        if len(keys) == 1:
            return next(iter(keys))
        merged_spec: _TempStorageCtorSpec | None = None
        ordered_keys = sorted(
            keys, key=lambda key: (self._temp_storage_ctor_order.get(key, 1 << 30), key)
        )
        for key in ordered_keys:
            spec = self._temp_storage_ctor_specs.get(key)
            if spec is None:
                continue
            merged_spec = self._merge_temp_storage_ctor_specs(merged_spec, spec)
        if merged_spec is None:
            return None
        canonical_key = ordered_keys[0]
        self._temp_storage_ctor_specs[canonical_key] = merged_spec
        alias_orders = [
            self._temp_storage_ctor_order[key]
            for key in ordered_keys
            if key in self._temp_storage_ctor_order
        ]
        if alias_orders:
            self._temp_storage_ctor_order[canonical_key] = min(alias_orders)
        return canonical_key

    def _resolve_temp_storage_plan(self, value: ir.Var) -> _TempStoragePlan | None:
        key = self._resolve_temp_storage_ctor_key(value)
        if key is None:
            return None
        if self._temp_storage_global_plan is None and self._temp_storage_ctor_specs:
            self._ensure_temp_storage_global_plan()
        return self._finalize_temp_storage_plan_for_var(key)

    def _finalize_temp_storage_plan_for_var(self, var_name: str) -> _TempStoragePlan:
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
        required_alignment = (
            max(_MIN_TEMP_STORAGE_ALIGNMENT, *(entry.alignment for entry in uses))
            if uses
            else max(_MIN_TEMP_STORAGE_ALIGNMENT, int(ctor_spec.alignment or 1))
        )
        slices_by_call_id: dict[int, _TempStorageSlice] = {}
        if ctor_spec.sharing == "shared":
            required_size = max((entry.size_in_bytes for entry in uses), default=0)
            for entry in uses:
                slices_by_call_id[id(entry.call_assign)] = _TempStorageSlice(
                    offset=0, size_in_bytes=entry.size_in_bytes
                )
        else:
            required_size = 0
            for entry in uses:
                required_size = _align_up(required_size, max(1, int(entry.alignment)))
                slices_by_call_id[id(entry.call_assign)] = _TempStorageSlice(
                    offset=required_size, size_in_bytes=entry.size_in_bytes
                )
                required_size += entry.size_in_bytes
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
            except _INFERENCE_EXCEPTIONS:
                func_obj = None
            if func_obj is None:
                func_def = self._lookup_definition(func_ref)
                if isinstance(func_def, (ir.Global, ir.FreeVar, ir.Const)):
                    func_obj = func_def.value
        elif isinstance(func_ref, (ir.Global, ir.FreeVar, ir.Const)):
            func_obj = func_ref.value
        if func_obj is None:
            chain = self._resolve_attribute_chain(func_ref)
            if chain is not None:
                root, attrs = chain
                obj = root
                try:
                    for attr in attrs:
                        obj = getattr(obj, attr)
                    func_obj = obj
                except _INFERENCE_EXCEPTIONS:
                    func_obj = None
        if func_obj is None:
            return None
        try:
            return np.dtype(func_obj).type
        except (TypeError, ValueError):
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

    def _infer_thread_data_dtype_from_provenance_writes(self, value: ir.Var):
        """Infer dtype from writes without re-entering marker spec resolution."""
        inferred = None
        roots = self._collect_thread_data_write_roots(value)
        for root_name in sorted(roots):
            root = roots[root_name]
            spec = self._resolve_thread_data_spec(root)
            root_dtype = spec.dtype if spec is not None else None
            if root_dtype is None:
                root_dtype = self._resolve_var_dtype(root)
            if root_dtype is None:
                root_dtype = self._infer_thread_data_dtype_from_writes(root)
            if root_dtype is None:
                continue
            if inferred is not None and not _dtype_values_match(inferred, root_dtype):
                raise CoopSinglePhaseRewriteError(
                    "Inconsistent inferred dtype across coop.ThreadData "
                    "payload provenance."
                )
            inferred = root_dtype
        return inferred

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
        definition = self._lookup_definition(value)
        if isinstance(definition, ir.Const):
            try:
                return _numba_typeof(definition.value)
            except (TypeError, ValueError):
                return None
        if isinstance(definition, ir.Var):
            return self._resolve_var_dtype(definition, seen)
        if isinstance(definition, ir.Expr) and definition.op == "getattr":
            if definition.attr == "dtype" and isinstance(definition.value, ir.Var):
                return self._resolve_var_dtype(definition.value, seen)
        if isinstance(definition, ir.Expr) and definition.op in {
            "getitem",
            "static_getitem",
        }:
            base_value = getattr(definition, "value", None)
            if isinstance(base_value, ir.Var):
                return self._resolve_var_dtype(base_value, seen)
        if isinstance(definition, ir.Expr) and definition.op in {
            "binop",
            "inplace_binop",
        }:
            lhs = getattr(definition, "lhs", None)
            rhs = getattr(definition, "rhs", None)
            lhs_dtype = (
                self._resolve_var_dtype(lhs, seen) if isinstance(lhs, ir.Var) else None
            )
            rhs_dtype = (
                self._resolve_var_dtype(rhs, seen) if isinstance(rhs, ir.Var) else None
            )
            if (
                lhs_dtype is not None
                and rhs_dtype is not None
                and (lhs_dtype != rhs_dtype)
            ):
                return None
            return lhs_dtype if lhs_dtype is not None else rhs_dtype
        if isinstance(definition, ir.Expr) and definition.op == "unary":
            unary_value = getattr(definition, "value", None)
            if isinstance(unary_value, ir.Var):
                return self._resolve_var_dtype(unary_value, seen)
        if isinstance(definition, ir.Expr) and definition.op == "phi":
            inferred = None
            for incoming in _phi_incoming_values(definition):
                if not isinstance(incoming, ir.Var):
                    continue
                incoming_dtype = self._resolve_var_dtype(incoming, seen)
                if incoming_dtype is None:
                    continue
                if inferred is None:
                    inferred = incoming_dtype
                    continue
                if inferred != incoming_dtype:
                    return None
            if inferred is not None:
                return inferred
        if isinstance(definition, ir.Expr) and definition.op == "call":
            call_dtype = self._resolve_call_result_dtype(definition)
            if call_dtype is not None:
                return call_dtype
        return None

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

    def _resolve_factory_kwarg_value(self, name: str, value_ref):
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
        if name in {
            "dtype",
            "item_dtype",
            "counter_dtype",
            "run_length_dtype",
            "decoded_offset_dtype",
            "total_decoded_size_dtype",
            "relative_offset_dtype",
        }:
            dtype = self._resolve_dtype_ref(value_ref)
            if dtype is not None:
                return dtype
        return _UNRESOLVED

    def _resolve_call_target(self, call: ir.Expr):
        factory = self._resolve_factory_from_var(call.func)
        if factory is not None:
            metadata = factory_operation(factory)
            assert metadata is not None
            return _ResolvedCallTarget(
                factory=factory,
                operation=metadata.operation,
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
            operation=metadata.operation,
            func_var_name=call.func.name,
            func_var_name_extra=func_def.value.name,
            getitem_temp_storage=getitem_temp_storage,
        )


__all__ = ["_ProvenanceRewrite"]
