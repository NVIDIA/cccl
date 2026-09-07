# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Temporary-storage layout and IR emission.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from ._rewrite_support import (
    _DEFAULT_STATIC_SHARED_MEMORY_BYTES,
    _GLOBAL_NAME_COUNTER,
    CoopSinglePhaseRewriteError,
    _align_up,
    _cuda_module,
    _next_global_name,
    _query_device_shared_memory_limits,
    _RewriteMatch,
    _TempStorageGlobalPlan,
    _TempStoragePlan,
    _TempStorageRequirementSummary,
    _TempStorageUseRequirement,
    ir,
    normalize_dim_param,
    operator,
    replace,
    set_required_dynamic_shared_memory,
)


class _StorageRewrite:
    def _get_device_shared_memory_limits(self) -> tuple[int, int]:
        max_default = _DEFAULT_STATIC_SHARED_MEMORY_BYTES
        max_optin = max_default
        try:
            limits = _query_device_shared_memory_limits()
            max_default = int(
                limits.get("max_default_shared_memory_per_block", max_default)
                or max_default
            )
            max_optin = int(
                limits.get("max_optin_shared_memory_per_block", max_default)
                or max_default
            )
            if max_optin <= 0:
                max_optin = max_default
        except (AttributeError, KeyError, OSError, RuntimeError, TypeError, ValueError):
            pass
        return (max_default, max_optin)

    def _ensure_temp_storage_global_plan(self) -> _TempStorageGlobalPlan:
        cached = self._temp_storage_global_plan
        if cached is not None:
            return cached
        ordered_keys = sorted(
            self._temp_storage_ctor_specs.keys(),
            key=lambda name: (self._temp_storage_ctor_order.get(name, 1 << 30), name),
        )
        offset = 0
        max_alignment = 1
        for key in ordered_keys:
            plan = self._finalize_temp_storage_plan_for_var(key)
            alignment = max(1, int(plan.alignment))
            offset = _align_up(offset, alignment)
            self._temp_storage_plans[key] = replace(plan, base_offset=offset)
            offset += int(plan.size_in_bytes)
            max_alignment = max(max_alignment, alignment)
        total_size = _align_up(offset, max_alignment)
        max_default, max_optin = self._get_device_shared_memory_limits()
        uses_dynamic_smem = total_size > max_default
        dynamic_shared_bytes = total_size if uses_dynamic_smem else 0
        if dynamic_shared_bytes > max_optin:
            raise CoopSinglePhaseRewriteError(
                f"TempStorage requires {dynamic_shared_bytes} bytes dynamic shared memory, but device max opt-in is {max_optin} bytes."
            )
        if dynamic_shared_bytes > 0:
            set_required_dynamic_shared_memory(self._state, dynamic_shared_bytes)
        plan = _TempStorageGlobalPlan(
            total_size=total_size,
            max_alignment=max_alignment,
            uses_dynamic_smem=uses_dynamic_smem,
            dynamic_shared_bytes=dynamic_shared_bytes,
            max_default_smem=max_default,
            max_optin_smem=max_optin,
        )
        self._temp_storage_global_plan = plan
        return plan

    def _emit_array_slice(
        self,
        block: ir.Block,
        *,
        source_var: ir.Var,
        target_var: ir.Var,
        start: int,
        stop: int,
        loc: ir.Loc,
    ) -> None:
        slice_ctor_global_name = _next_global_name("temp_storage_slice_ctor")
        slice_ctor_var = ir.Var(
            target_var.scope,
            f"__coop_temp_storage_slice_ctor_var_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        start_var = ir.Var(
            target_var.scope,
            f"__coop_temp_storage_slice_start_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        stop_var = ir.Var(
            target_var.scope,
            f"__coop_temp_storage_slice_stop_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        slice_obj_var = ir.Var(
            target_var.scope,
            f"__coop_temp_storage_slice_obj_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        block.append(
            ir.Assign(
                ir.Global(slice_ctor_global_name, slice, loc), slice_ctor_var, loc
            )
        )
        block.append(ir.Assign(ir.Const(int(start), loc), start_var, loc))
        block.append(ir.Assign(ir.Const(int(stop), loc), stop_var, loc))
        block.append(
            ir.Assign(
                ir.Expr.call(slice_ctor_var, [start_var, stop_var], (), loc),
                slice_obj_var,
                loc,
            )
        )
        block.append(
            ir.Assign(ir.Expr.getitem(source_var, slice_obj_var, loc), target_var, loc)
        )

    def _runtime_temp_storage_arg_for_call(
        self, block: ir.Block, *, source_var: ir.Var, call_assign: ir.Assign
    ) -> tuple[ir.Var, _TempStoragePlan | None]:
        temp_storage_arg = source_var
        temp_storage_plan = self._resolve_temp_storage_plan(source_var)
        if temp_storage_plan is not None and temp_storage_plan.sharing == "exclusive":
            slice_info = temp_storage_plan.slices_by_call_id.get(id(call_assign))
            if slice_info is None:
                raise CoopSinglePhaseRewriteError(
                    f"Could not resolve TempStorage slice for call at {call_assign.loc}."
                )
            sliced_var = ir.Var(
                call_assign.target.scope,
                f"__coop_temp_storage_slice_{next(_GLOBAL_NAME_COUNTER)}__",
                call_assign.loc,
            )
            self._emit_array_slice(
                block,
                source_var=source_var,
                target_var=sliced_var,
                start=slice_info.offset,
                stop=slice_info.offset + slice_info.size_in_bytes,
                loc=call_assign.loc,
            )
            temp_storage_arg = sliced_var
        return (temp_storage_arg, temp_storage_plan)

    def _emit_temp_storage_auto_sync(
        self, block: ir.Block, *, scope: ir.Scope, loc: ir.Loc, sync_attr: str
    ) -> None:
        sync_module_global_name = _next_global_name("temp_storage_sync_mod")
        sync_module_var = ir.Var(
            scope, f"__coop_sync_mod_var_{next(_GLOBAL_NAME_COUNTER)}__", loc
        )
        sync_fn_var = ir.Var(
            scope, f"__coop_sync_fn_{next(_GLOBAL_NAME_COUNTER)}__", loc
        )
        sync_result_var = ir.Var(
            scope, f"__coop_sync_result_{next(_GLOBAL_NAME_COUNTER)}__", loc
        )
        block.append(
            ir.Assign(
                ir.Global(sync_module_global_name, _cuda_module, loc),
                sync_module_var,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.getattr(sync_module_var, sync_attr, loc), sync_fn_var, loc
            )
        )
        block.append(
            ir.Assign(ir.Expr.call(sync_fn_var, (), (), loc), sync_result_var, loc)
        )

    def _emit_physical_warp_tile_offset(
        self,
        block: ir.Block,
        *,
        match: _RewriteMatch,
        user_offset: ir.Var,
        scope: ir.Scope,
        loc: ir.Loc,
    ) -> ir.Var:
        items_per_thread = match.factory_kwargs.get("items_per_thread")
        if (
            not isinstance(items_per_thread, int)
            or isinstance(items_per_thread, bool)
            or items_per_thread < 1
        ):
            raise CoopSinglePhaseRewriteError(
                "root physical-warp load/store requires an inferred positive items_per_thread"
            )
        logical_warp_threads = match.factory_kwargs.get("threads_in_warp", 32)
        if (
            not isinstance(logical_warp_threads, int)
            or isinstance(logical_warp_threads, bool)
            or logical_warp_threads < 1
            or (32 % logical_warp_threads != 0)
        ):
            raise CoopSinglePhaseRewriteError(
                "root warp load/store requires threads_in_warp to be a positive divisor of 32"
            )
        block_dim = normalize_dim_param(match.factory_kwargs.get("threads_per_block"))

        def new_var(stem: str) -> ir.Var:
            return ir.Var(
                scope, f"__coop_warp_tile_{stem}_{next(_GLOBAL_NAME_COUNTER)}__", loc
            )

        def constant(value: int, stem: str) -> ir.Var:
            result = new_var(stem)
            block.append(ir.Assign(ir.Const(value, loc), result, loc))
            return result

        def binary(function, lhs: ir.Var, rhs: ir.Var, stem: str) -> ir.Var:
            result = new_var(stem)
            block.append(ir.Assign(ir.Expr.binop(function, lhs, rhs, loc), result, loc))
            return result

        module_var = new_var("cuda")
        block.append(
            ir.Assign(
                ir.Global(_next_global_name("warp_tile_cuda"), _cuda_module, loc),
                module_var,
                loc,
            )
        )
        thread_idx = new_var("thread_idx")
        block.append(
            ir.Assign(ir.Expr.getattr(module_var, "threadIdx", loc), thread_idx, loc)
        )

        def component(axis: str) -> ir.Var:
            result = new_var(f"thread_idx_{axis}")
            block.append(ir.Assign(ir.Expr.getattr(thread_idx, axis, loc), result, loc))
            return result

        linear_tid = component("x")
        if block_dim[1] > 1 or block_dim[2] > 1:
            y = component("y")
            z = component("z")
            yz = binary(operator.mul, constant(block_dim[1], "block_y"), z, "linear_yz")
            yz = binary(operator.add, y, yz, "linear_y")
            yz = binary(
                operator.mul, constant(block_dim[0], "block_x"), yz, "linear_x_stride"
            )
            linear_tid = binary(operator.add, linear_tid, yz, "linear_tid")
        warp_id = binary(
            operator.floordiv,
            linear_tid,
            constant(logical_warp_threads, "warp_threads"),
            "warp_id",
        )
        tile_offset = binary(
            operator.mul,
            warp_id,
            constant(logical_warp_threads * items_per_thread, "warp_tile_items"),
            "implicit_offset",
        )
        return binary(operator.add, tile_offset, user_offset, "offset")

    def _emit_root_store_payload(
        self,
        block: ir.Block,
        *,
        match: _RewriteMatch,
        value: ir.Var,
        scope: ir.Scope,
        loc: ir.Loc,
    ) -> ir.Var:
        dtype = match.factory_kwargs.get("dtype")
        if dtype is None:
            raise CoopSinglePhaseRewriteError("root store requires an inferred dtype")
        items_per_thread = (
            1
            if match.root_store_scalar
            else match.factory_kwargs.get("items_per_thread")
        )
        if (
            isinstance(items_per_thread, bool)
            or not isinstance(items_per_thread, int)
            or items_per_thread < 1
        ):
            raise CoopSinglePhaseRewriteError(
                "root store requires an inferred positive items_per_thread"
            )

        def new_var(stem: str) -> ir.Var:
            return ir.Var(
                scope, f"__coop_root_store_{stem}_{next(_GLOBAL_NAME_COUNTER)}__", loc
            )

        module_var = new_var("cuda")
        local_var = new_var("local")
        array_fn = new_var("array")
        shape_var = new_var("shape")
        dtype_var = new_var("dtype")
        payload = new_var("payload")
        block.append(
            ir.Assign(
                ir.Global(_next_global_name("root_store_cuda"), _cuda_module, loc),
                module_var,
                loc,
            )
        )
        block.append(
            ir.Assign(ir.Expr.getattr(module_var, "local", loc), local_var, loc)
        )
        block.append(ir.Assign(ir.Expr.getattr(local_var, "array", loc), array_fn, loc))
        block.append(ir.Assign(ir.Const(items_per_thread, loc), shape_var, loc))
        block.append(
            ir.Assign(
                ir.Global(_next_global_name("root_store_dtype"), dtype, loc),
                dtype_var,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.call(array_fn, [shape_var, dtype_var], (), loc), payload, loc
            )
        )
        for item_index in range(items_per_thread):
            index_var = new_var(f"index_{item_index}")
            block.append(ir.Assign(ir.Const(item_index, loc), index_var, loc))
            item_var = value
            if not match.root_store_scalar:
                item_var = new_var(f"item_{item_index}")
                block.append(
                    ir.Assign(ir.Expr.getitem(value, index_var, loc), item_var, loc)
                )
            block.append(ir.SetItem(payload, index_var, item_var, loc))
        return payload

    def _compute_func_temp_storage_requirements(
        self, func_ir
    ) -> dict[str, _TempStorageRequirementSummary]:
        requirements: dict[str, _TempStorageRequirementSummary] = {}
        saved_block_defs = self._block_defs
        saved_block = self._block
        self._temp_storage_ctor_specs = {}
        self._temp_storage_ctor_order = {}
        try:
            ctor_order = 0
            for label in sorted(func_ir.blocks):
                scan_block = func_ir.blocks[label]
                self._block = scan_block
                self._block_defs = {
                    inst.target.name: inst.value
                    for inst in scan_block.body
                    if isinstance(inst, ir.Assign)
                }
                for inst in scan_block.body:
                    if not isinstance(inst, ir.Assign):
                        continue
                    call = inst.value
                    if not isinstance(call, ir.Expr) or call.op != "call":
                        continue
                    if self._is_thread_data_ctor_call(call):
                        self._thread_data_specs[inst.target.name] = (
                            self._merge_thread_data_specs(
                                self._thread_data_specs.get(inst.target.name),
                                self._extract_thread_data_spec(call),
                            )
                        )
                    elif self._is_typed_group_payload_ctor_call(call):
                        self._thread_data_specs[inst.target.name] = (
                            self._merge_thread_data_specs(
                                self._thread_data_specs.get(inst.target.name),
                                self._extract_typed_group_payload_spec(call),
                            )
                        )
                    elif self._is_temp_storage_ctor_call(call):
                        self._temp_storage_ctor_specs[inst.target.name] = (
                            self._extract_temp_storage_ctor_spec(call)
                        )
                        self._temp_storage_ctor_order[inst.target.name] = ctor_order
                        ctor_order += 1
            all_scan_matches: list[_RewriteMatch] = []
            storage_uses: list[tuple[int, ir.Assign, _RewriteMatch, str]] = []
            source_order = 0
            for label in sorted(func_ir.blocks):
                scan_block = func_ir.blocks[label]
                self._block = scan_block
                self._block_defs = {
                    inst.target.name: inst.value
                    for inst in scan_block.body
                    if isinstance(inst, ir.Assign)
                }
                for inst in scan_block.body:
                    current_order = source_order
                    source_order += 1
                    if not isinstance(inst, ir.Assign):
                        continue
                    call = inst.value
                    if not isinstance(call, ir.Expr) or call.op != "call":
                        continue
                    target = self._resolve_call_target(call)
                    if target is None:
                        continue
                    op_name = target.operation
                    (
                        runtime_args,
                        runtime_temp_storage_var,
                        factory_kwargs,
                        factory_kw_value_vars,
                    ) = self._validate_and_split_args(
                        op_name, call, target.getitem_temp_storage
                    )
                    (
                        physical_warp_tile_origin,
                        preserve_root_store_payload,
                        root_store_scalar,
                    ) = self._extract_group_root_match_metadata(
                        op_name=op_name,
                        runtime_args=runtime_args,
                        factory_kwargs=factory_kwargs,
                    )
                    scan_match = _RewriteMatch(
                        op_name=op_name,
                        factory=target.factory,
                        func_var_name=target.func_var_name,
                        func_var_name_extra=target.func_var_name_extra,
                        runtime_args=runtime_args,
                        runtime_temp_storage_var=runtime_temp_storage_var,
                        factory_kwargs=factory_kwargs,
                        factory_kw_value_vars=factory_kw_value_vars,
                        loc=inst.loc,
                        physical_warp_tile_origin=physical_warp_tile_origin,
                        preserve_root_store_payload=preserve_root_store_payload,
                        root_store_scalar=root_store_scalar,
                    )
                    all_scan_matches.append(scan_match)
                    if runtime_temp_storage_var is None:
                        continue
                    ctor_key = self._resolve_temp_storage_ctor_key(
                        runtime_temp_storage_var
                    )
                    if ctor_key is not None:
                        storage_uses.append((current_order, inst, scan_match, ctor_key))
            self._prepare_ltoir_bundle_for_matches(all_scan_matches)
            for source_order, inst, scan_match, ctor_key in storage_uses:
                invocable, _ = self._materialize_invocable(scan_match)
                size_in_bytes = max(
                    1, int(getattr(invocable, "temp_storage_bytes", 0) or 0)
                )
                alignment = max(
                    1, int(getattr(invocable, "temp_storage_alignment", 0) or 0)
                )
                summary = requirements.setdefault(
                    ctor_key, _TempStorageRequirementSummary()
                )
                summary.max_size_in_bytes = max(
                    summary.max_size_in_bytes, size_in_bytes
                )
                summary.max_alignment = max(summary.max_alignment, alignment)
                summary.uses.append(
                    _TempStorageUseRequirement(
                        call_assign=inst,
                        order=source_order,
                        size_in_bytes=size_in_bytes,
                        alignment=alignment,
                    )
                )
        finally:
            self._block_defs = saved_block_defs
            self._block = saved_block
        return requirements


__all__ = ["_StorageRewrite"]
