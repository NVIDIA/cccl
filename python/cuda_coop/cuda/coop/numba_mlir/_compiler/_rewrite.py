# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Single-phase rewrite orchestration for Numba-CUDA-MLIR.

The concrete analysis and primitive finalization responsibilities are split
across focused rewrite mixins. This module owns registration, the stable
operation specification table, match/apply ordering, and whole-function retry.
"""

from ._operations import StorageABI
from ._rewrite_arguments import _ArgumentRewrite
from ._rewrite_group_metadata import _GroupMetadataRewrite
from ._rewrite_invocables import _InvocableRewrite
from ._rewrite_launch import _LaunchRewrite
from ._rewrite_payload import _PayloadRewrite
from ._rewrite_provenance import _ProvenanceRewrite
from ._rewrite_storage import _StorageRewrite
from ._rewrite_support import (
    _GLOBAL_NAME_COUNTER,
    CoopSinglePhaseRewriteError,
    Rewrite,
    WholeFunctionPlanner,
    _cuda_module,
    _DeferredCoopRewrite,
    _next_global_name,
    _RewriteMatch,
    ir,
    register_planner,
    register_rewrite,
    require_launch_config,
)


@register_rewrite("before-inference")
class CoopSinglePhaseRewrite(
    _ProvenanceRewrite,
    _ArgumentRewrite,
    _LaunchRewrite,
    _GroupMetadataRewrite,
    _PayloadRewrite,
    _InvocableRewrite,
    _StorageRewrite,
    Rewrite,
):
    """Rewrite planner-private providers into two-phase invocable calls."""

    def match(self, func_ir, block, typemap, calltypes):
        from ._group_planner import has_group_markers

        if has_group_markers(func_ir):
            return False
        func_ir_identity = id(func_ir)
        if self._func_ir_identity != func_ir_identity:
            self._func_ir_identity = func_ir_identity
            self._func_ir = func_ir
            self._thread_data_specs = {}
            self._temp_storage_plans = {}
            self._temp_storage_global_plan = None
            self._temp_storage_ctor_order = {}
            self._temp_storage_ctor_roots = {}
            self._implicit_temp_storage_plan = None
            self._temp_storage_backing_var = None
            self._temp_storage_backing_emitted = False
            self._prebundled_specializations = {}
            try:
                self._func_temp_storage_requirements = (
                    self._compute_func_temp_storage_requirements(func_ir)
                )
            except _DeferredCoopRewrite:
                self._func_temp_storage_requirements = {}
                return False
        self._block = block
        self._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        self._matches = {}
        self._temp_storage_assigns = set()
        self._temp_storage_func_vars = set()
        self._thread_data_func_vars = set()
        self._typed_group_payload_func_vars = set()
        for inst in block.body:
            if not isinstance(inst, ir.Assign):
                continue
            call = inst.value
            if not isinstance(call, ir.Expr) or call.op != "call":
                continue
            if self._is_temp_storage_ctor_call(call):
                self._temp_storage_assigns.add(inst)
                self._temp_storage_func_vars.add(call.func.name)
                self._temp_storage_ctor_specs[inst.target.name] = (
                    self._extract_temp_storage_ctor_spec(call)
                )
                self._temp_storage_ctor_order.setdefault(
                    inst.target.name, len(self._temp_storage_ctor_order)
                )
                continue
            if self._is_thread_data_ctor_call(call):
                self._thread_data_func_vars.add(call.func.name)
                self._thread_data_specs[inst.target.name] = (
                    self._merge_thread_data_specs(
                        self._thread_data_specs.get(inst.target.name),
                        self._extract_thread_data_spec(call),
                    )
                )
                continue
            if self._is_typed_group_payload_ctor_call(call):
                self._typed_group_payload_func_vars.add(call.func.name)
                self._thread_data_specs[inst.target.name] = (
                    self._merge_thread_data_specs(
                        self._thread_data_specs.get(inst.target.name),
                        self._extract_typed_group_payload_spec(call),
                    )
                )
                continue
            target = self._resolve_call_target(call)
            if target is None:
                continue
            op_name = target.operation
            try:
                (
                    runtime_args,
                    runtime_temp_storage_var,
                    factory_kwargs,
                    factory_kw_value_vars,
                ) = self._validate_and_split_args(
                    op_name, call, target.getitem_temp_storage
                )
            except _DeferredCoopRewrite:
                continue
            family_metadata = self._analyze_family_match(
                op_name=op_name,
                runtime_args=runtime_args,
                factory_kwargs=factory_kwargs,
            )
            self._matches[inst] = _RewriteMatch(
                op_name=op_name,
                factory=target.factory,
                factory_metadata=target.factory_metadata,
                func_var_name=target.func_var_name,
                func_var_name_extra=target.func_var_name_extra,
                runtime_args=runtime_args,
                runtime_temp_storage_var=runtime_temp_storage_var,
                factory_kwargs=factory_kwargs,
                factory_kw_value_vars=factory_kw_value_vars,
                loc=inst.loc,
                family_metadata=family_metadata,
            )
        if self._deferred_launch_dim_inference:
            # Keep all helper constructors and launch-dependent calls intact.
            # A kernel planner will request exact launch metadata and retry the
            # complete function with a fresh rewrite object. Device-function
            # markers remain available for rewriting after caller inlining.
            return False

        return (
            bool(self._matches)
            or bool(self._temp_storage_assigns)
            or bool(self._thread_data_func_vars)
            or bool(self._typed_group_payload_func_vars)
        )

    def apply(self):
        assert self._block is not None
        call_invocable_globals: dict[ir.Assign, tuple[str, object]] = {}
        func_var_names_to_clear: set[str] = set()
        candidate_dead_factory_kw_vars: set[str] = set()
        if self._has_temp_storage_requirements():
            self._stage_temp_storage_backing()
        for match_inst, match in self._matches.items():
            invocable, _ = self._materialize_invocable(match)
            self._record_invocable_specialization(invocable)
            candidate_dead_factory_kw_vars.update(
                (value_var.name for value_var in match.factory_kw_value_vars)
            )
            global_name = _next_global_name("single_phase")
            call_invocable_globals[match_inst] = (global_name, invocable)
            func_var_names_to_clear.add(match.func_var_name)
            if match.func_var_name_extra is not None:
                func_var_names_to_clear.add(match.func_var_name_extra)
        new_block = ir.Block(self._block.scope, self._block.loc)
        for inst in self._block.body:
            if (
                isinstance(inst, ir.Assign)
                and inst.target.name
                in self._thread_data_func_vars | self._typed_group_payload_func_vars
            ):
                module_var = ir.Var(
                    inst.target.scope,
                    f"__coop_thread_data_module_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                local_module_var = ir.Var(
                    inst.target.scope,
                    f"__coop_thread_data_local_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                new_block.append(
                    ir.Assign(
                        ir.Global(
                            _next_global_name("thread_data_module"),
                            _cuda_module,
                            inst.loc,
                        ),
                        module_var,
                        inst.loc,
                    )
                )
                new_block.append(
                    ir.Assign(
                        ir.Expr.getattr(module_var, "local", inst.loc),
                        local_module_var,
                        inst.loc,
                    )
                )
                new_block.append(
                    ir.Assign(
                        ir.Expr.getattr(local_module_var, "array", inst.loc),
                        inst.target,
                        inst.loc,
                    )
                )
                continue
            if (
                isinstance(inst, ir.Assign)
                and inst.target.name in func_var_names_to_clear
            ):
                new_block.append(
                    ir.Assign(ir.Const(None, inst.loc), inst.target, inst.loc)
                )
                continue
            if (
                isinstance(inst, ir.Assign)
                and inst.target.name in self._temp_storage_func_vars
            ):
                new_block.append(
                    ir.Assign(ir.Const(None, inst.loc), inst.target, inst.loc)
                )
                continue
            if (
                isinstance(inst, ir.Assign)
                and isinstance(inst.value, ir.Expr)
                and (inst.value.op == "call")
                and (
                    self._is_thread_data_ctor_call(inst.value)
                    or self._is_typed_group_payload_ctor_call(inst.value)
                )
            ):
                is_typed_group_payload = self._is_typed_group_payload_ctor_call(
                    inst.value
                )
                thread_data_spec = self._thread_data_specs.get(inst.target.name)
                if thread_data_spec is not None and thread_data_spec.dtype is None:
                    self._infer_thread_data_dtype_from_writes(inst.target)
                    thread_data_spec = self._thread_data_specs.get(inst.target.name)
                if thread_data_spec is None or thread_data_spec.dtype is None:
                    subject = (
                        "typed group payload"
                        if is_typed_group_payload
                        else "coop.ThreadData(...)"
                    )
                    raise CoopSinglePhaseRewriteError(
                        f"Failed to infer dtype for {subject}. Use it with a "
                        "cooperative group operation that provides dtype context."
                    )
                if thread_data_spec.common_root:
                    from ._parameters import _validate_common_numeric_dtype

                    try:
                        _validate_common_numeric_dtype(
                            thread_data_spec.dtype, operation="ThreadData"
                        )
                    except (TypeError, ValueError) as exc:
                        raise CoopSinglePhaseRewriteError(str(exc)) from exc
                dtype_var = ir.Var(
                    inst.target.scope,
                    f"__coop_thread_data_dtype_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                new_block.append(
                    ir.Assign(
                        ir.Global(
                            _next_global_name("thread_data_dtype"),
                            thread_data_spec.dtype,
                            inst.loc,
                        ),
                        dtype_var,
                        inst.loc,
                    )
                )
                rewritten_args = [] if is_typed_group_payload else list(inst.value.args)
                rewritten_kws = [] if is_typed_group_payload else list(inst.value.kws)
                rewritten_kws = [
                    ("shape" if name == "items_per_thread" else name, value)
                    for name, value in rewritten_kws
                    if name != "alignas"
                ]
                if thread_data_spec.items_per_thread is None:
                    raise CoopSinglePhaseRewriteError(
                        "Failed to infer static extent for typed group payload."
                    )
                items_var = ir.Var(
                    inst.target.scope,
                    f"__coop_thread_data_items_{next(_GLOBAL_NAME_COUNTER)}__",
                    inst.loc,
                )
                new_block.append(
                    ir.Assign(
                        ir.Const(thread_data_spec.items_per_thread, inst.loc),
                        items_var,
                        inst.loc,
                    )
                )
                if rewritten_args:
                    rewritten_args[0] = items_var
                elif any((name == "shape" for name, _ in rewritten_kws)):
                    rewritten_kws = [
                        (name, items_var if name == "shape" else value)
                        for name, value in rewritten_kws
                    ]
                else:
                    rewritten_args.append(items_var)
                if len(rewritten_args) >= 2:
                    rewritten_args[1] = dtype_var
                elif any((name == "dtype" for name, _ in rewritten_kws)):
                    rewritten_kws = [
                        (name, dtype_var if name == "dtype" else value)
                        for name, value in rewritten_kws
                    ]
                elif rewritten_args:
                    rewritten_args.append(dtype_var)
                else:
                    rewritten_kws.append(("dtype", dtype_var))
                if thread_data_spec.alignment is not None:
                    alignment_var = ir.Var(
                        inst.target.scope,
                        f"__coop_thread_data_alignment_{next(_GLOBAL_NAME_COUNTER)}__",
                        inst.loc,
                    )
                    new_block.append(
                        ir.Assign(
                            ir.Const(thread_data_spec.alignment, inst.loc),
                            alignment_var,
                            inst.loc,
                        )
                    )
                    rewritten_kws.append(("alignment", alignment_var))
                new_block.append(
                    ir.Assign(
                        ir.Expr.call(
                            inst.value.func,
                            rewritten_args,
                            tuple(rewritten_kws),
                            inst.loc,
                        ),
                        inst.target,
                        inst.loc,
                    )
                )
                continue
            match = self._matches.get(inst)
            if match is None and inst not in self._temp_storage_assigns:
                new_block.append(inst)
                continue
            if inst in self._temp_storage_assigns:
                ctor_key = self._resolve_temp_storage_ctor_key(inst.target)
                if ctor_key is None:
                    raise CoopSinglePhaseRewriteError(
                        f"Missing TempStorage metadata for '{inst.target.name}'."
                    )
                if ctor_key not in self._func_temp_storage_requirements:
                    new_block.append(
                        ir.Assign(ir.Const(None, inst.loc), inst.target, inst.loc)
                    )
                    continue
                plan = self._finalize_temp_storage_plan_for_var(ctor_key)
                backing_var = self._temp_storage_backing_var
                if backing_var is None:
                    raise CoopSinglePhaseRewriteError(
                        "Missing unified TempStorage backing allocation."
                    )
                self._emit_array_slice(
                    new_block,
                    source_var=backing_var,
                    target_var=inst.target,
                    start=plan.base_offset,
                    stop=plan.base_offset + plan.size_in_bytes,
                    loc=inst.loc,
                )
                continue
            assert match is not None
            rewritten_runtime_args = self._prepare_family_runtime_args(
                new_block,
                match=match,
                runtime_args=list(match.runtime_args),
                scope=inst.target.scope,
                loc=match.loc,
            )
            runtime_temp_storage_plan = None
            if match.factory_metadata.storage_abi is StorageABI.LEADING_POINTER:
                if match.runtime_temp_storage_var is not None:
                    runtime_temp_storage_arg, runtime_temp_storage_plan = (
                        self._runtime_temp_storage_arg_for_call(
                            new_block,
                            source_var=match.runtime_temp_storage_var,
                            call_assign=inst,
                        )
                    )
                else:
                    (
                        runtime_temp_storage_arg,
                        runtime_temp_storage_plan,
                    ) = self._implicit_temp_storage_arg_for_call(
                        new_block,
                        call_assign=inst,
                    )
                rewritten_runtime_args.insert(0, runtime_temp_storage_arg)
            call_func = inst.value.func
            call_invocable = call_invocable_globals.get(inst)
            if call_invocable is not None:
                global_name, invocable = call_invocable
                call_func = ir.Var(
                    inst.target.scope,
                    f"__coop_single_phase_call_{next(_GLOBAL_NAME_COUNTER)}__",
                    match.loc,
                )
                new_block.append(
                    ir.Assign(
                        ir.Global(global_name, invocable, match.loc),
                        call_func,
                        match.loc,
                    )
                )
            new_block.append(
                ir.Assign(
                    ir.Expr.call(call_func, rewritten_runtime_args, (), match.loc),
                    inst.target,
                    match.loc,
                )
            )
            if (
                runtime_temp_storage_plan is not None
                and runtime_temp_storage_plan.auto_sync
            ):
                self._emit_temp_storage_auto_sync(
                    new_block,
                    scope=inst.target.scope,
                    loc=inst.loc,
                    synchronization_scope=(
                        match.factory_metadata.synchronization_scope
                    ),
                )
        used_var_names: set[str] = set()
        for stmt in new_block.body:
            stmt_vars = list(stmt.list_vars())
            if isinstance(stmt, ir.Assign):
                stmt_vars = [var for var in stmt_vars if var.name != stmt.target.name]
            used_var_names.update((var.name for var in stmt_vars))
        if candidate_dead_factory_kw_vars:
            filtered_block = ir.Block(new_block.scope, new_block.loc)
            for stmt in new_block.body:
                if (
                    isinstance(stmt, ir.Assign)
                    and stmt.target.name in candidate_dead_factory_kw_vars
                    and (stmt.target.name not in used_var_names)
                ):
                    continue
                filtered_block.append(stmt)
            new_block = filtered_block
        self._state.typingctx.refresh()
        return new_block


from . import _group_planner  # noqa: E402, F401


@register_planner
class CoopWholeFunctionPlanner(WholeFunctionPlanner):
    """Apply cooperative-provider rewrites after device-function inlining."""

    def run(self) -> bool:
        rewrite = CoopSinglePhaseRewrite(self.state)
        modified = False

        def apply_matches() -> None:
            nonlocal modified
            for label in sorted(self.state.func_ir.blocks):
                block = self.state.func_ir.blocks[label]
                while rewrite.match(
                    self.state.func_ir,
                    block,
                    self.state.typemap,
                    self.state.calltypes,
                ):
                    block = rewrite.apply()
                    self.state.func_ir.blocks[label] = block
                    modified = True

        apply_matches()
        if rewrite._deferred_launch_dim_inference and not self.is_device_function:
            require_launch_config(self.state)
            rewrite = CoopSinglePhaseRewrite(
                self.state, allow_launch_dim_deferral=False
            )
            apply_matches()
        return modified


__all__ = [
    "CoopSinglePhaseRewrite",
    "CoopSinglePhaseRewriteError",
    "CoopWholeFunctionPlanner",
]
