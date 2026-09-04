# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Temporary-storage layout and IR emission.

This mixin is composed by CoopSinglePhaseRewrite. Registration and pass
ordering remain in the rewrite orchestrator.
"""

from cuda.coop._core import StorageOwnership, SynchronizationScope

from ._operations import _GROUP_LOWERING_PLAN_KWARG, StorageABI
from ._rewrite_support import (
    _DEFAULT_STATIC_SHARED_MEMORY_BYTES,
    _GLOBAL_NAME_COUNTER,
    CoopSinglePhaseRewriteError,
    _align_up,
    _cuda_module,
    _next_global_name,
    _phi_incoming_values,
    _query_device_shared_memory_limits,
    _RewriteMatch,
    _TempStorageGlobalPlan,
    _TempStoragePlan,
    _TempStorageRequirementSummary,
    _TempStorageSlice,
    _TempStorageUseRequirement,
    ir,
    operator,
    replace,
    set_required_dynamic_shared_memory,
)


class _StorageRewrite:
    @classmethod
    def _validate_storage_match_plan(cls, match: _RewriteMatch) -> None:
        lowering_plan = match.lowering_plan
        if lowering_plan is None:
            if (
                match.factory_metadata.execution_scope is not SynchronizationScope.BLOCK
                or match.factory_metadata.synchronization_scope
                is not SynchronizationScope.BLOCK
            ):
                raise CoopSinglePhaseRewriteError(
                    "storage-bearing providers without a group lowering plan "
                    "require block execution and block synchronization scopes"
                )
            return
        if lowering_plan.unsupported is not None:
            raise CoopSinglePhaseRewriteError(
                "cooperative provider storage received an unsupported "
                "group lowering plan."
            )
        topology = cls._validate_emittable_topology(lowering_plan)
        synchronization = lowering_plan.synchronization
        storage = lowering_plan.temp_storage
        if topology is None or synchronization is None or storage is None:
            raise CoopSinglePhaseRewriteError(
                "cooperative provider storage requires complete group "
                "topology, synchronization, and storage contracts."
            )
        if match.factory_metadata.execution_scope is not topology.execution_scope:
            raise CoopSinglePhaseRewriteError(
                "cooperative provider execution scope disagrees with its "
                "group topology."
            )
        if storage.ownership is StorageOwnership.NONE:
            raise CoopSinglePhaseRewriteError(
                "a storage-bearing cooperative provider received a "
                "storage-free lowering plan."
            )
        if storage.address_space != "shared":
            raise CoopSinglePhaseRewriteError(
                "storage-bearing cooperative providers require "
                "shared-address-space TempStorage."
            )
        if storage.instances != topology.instances or (
            storage.instance_index != topology.instance_index
        ):
            raise CoopSinglePhaseRewriteError(
                "cooperative provider storage layout disagrees with its group topology."
            )
        caller_owned = storage.ownership is StorageOwnership.CALLER
        if caller_owned != (match.runtime_temp_storage_var is not None):
            raise CoopSinglePhaseRewriteError(
                "cooperative provider TempStorage ownership disagrees with "
                "its runtime arguments."
            )
        if caller_owned and (
            topology.execution_scope is not SynchronizationScope.BLOCK
            or topology.instances != 1
        ):
            if topology.execution_scope is SynchronizationScope.WARP:
                raise CoopSinglePhaseRewriteError(
                    "cuda.coop.numba_mlir caller-owned TempStorage is not "
                    "supported for warp-scoped cooperative primitives; omit "
                    "temp_storage so the implementation can provide one "
                    "aligned slice per group instance"
                )
            raise CoopSinglePhaseRewriteError(
                "cuda.coop.numba_mlir caller-owned TempStorage is supported "
                "only for single-instance block-scoped cooperative primitives"
            )
        expected_reuse_barrier = (
            topology.execution_scope if storage.auto_sync else SynchronizationScope.NONE
        )
        planned_reuse_barrier = synchronization.storage_reuse_barrier
        if planned_reuse_barrier is not expected_reuse_barrier:
            raise CoopSinglePhaseRewriteError(
                "cooperative provider TempStorage automatic synchronization "
                "disagrees with its planned storage-reuse barrier."
            )
        allowed_provider_barriers = {planned_reuse_barrier}
        if caller_owned and not storage.auto_sync:
            allowed_provider_barriers.add(topology.execution_scope)
        if (
            match.factory_metadata.synchronization_scope
            not in allowed_provider_barriers
        ):
            raise CoopSinglePhaseRewriteError(
                "cooperative provider synchronization scope disagrees with "
                "its group lowering plan."
            )

    @staticmethod
    def _emit_integer_constant(
        block: ir.Block,
        *,
        scope: ir.Scope,
        loc: ir.Loc,
        stem: str,
        value: int,
    ) -> ir.Var:
        result = ir.Var(
            scope,
            f"__coop_{stem}_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        block.append(ir.Assign(ir.Const(int(value), loc), result, loc))
        return result

    @staticmethod
    def _emit_integer_binop(
        block: ir.Block,
        *,
        scope: ir.Scope,
        loc: ir.Loc,
        stem: str,
        fn,
        lhs: ir.Var,
        rhs: ir.Var,
    ) -> ir.Var:
        result = ir.Var(
            scope,
            f"__coop_{stem}_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        block.append(ir.Assign(ir.Expr.binop(fn, lhs, rhs, loc), result, loc))
        return result

    def _emit_linear_thread_rank(
        self,
        block: ir.Block,
        *,
        scope: ir.Scope,
        loc: ir.Loc,
    ) -> ir.Var:
        module_var = ir.Var(
            scope,
            f"__coop_group_topology_module_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        thread_idx_var = ir.Var(
            scope,
            f"__coop_group_topology_thread_idx_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        block_dim_var = ir.Var(
            scope,
            f"__coop_group_topology_block_dim_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        block.append(
            ir.Assign(
                ir.Global(
                    _next_global_name("group_topology_module"), _cuda_module, loc
                ),
                module_var,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.getattr(module_var, "threadIdx", loc), thread_idx_var, loc
            )
        )
        block.append(
            ir.Assign(ir.Expr.getattr(module_var, "blockDim", loc), block_dim_var, loc)
        )

        components = {}
        for aggregate_name, aggregate in (
            ("thread_idx", thread_idx_var),
            ("block_dim", block_dim_var),
        ):
            for component in ("x", "y", "z"):
                value = ir.Var(
                    scope,
                    f"__coop_group_topology_{aggregate_name}_{component}_"
                    f"{next(_GLOBAL_NAME_COUNTER)}__",
                    loc,
                )
                block.append(
                    ir.Assign(ir.Expr.getattr(aggregate, component, loc), value, loc)
                )
                components[aggregate_name, component] = value

        y_stride = self._emit_integer_binop(
            block,
            scope=scope,
            loc=loc,
            stem="group_topology_y_stride",
            fn=operator.mul,
            lhs=components["block_dim", "y"],
            rhs=components["thread_idx", "z"],
        )
        yz_rank = self._emit_integer_binop(
            block,
            scope=scope,
            loc=loc,
            stem="group_topology_yz_rank",
            fn=operator.add,
            lhs=components["thread_idx", "y"],
            rhs=y_stride,
        )
        x_stride = self._emit_integer_binop(
            block,
            scope=scope,
            loc=loc,
            stem="group_topology_x_stride",
            fn=operator.mul,
            lhs=components["block_dim", "x"],
            rhs=yz_rank,
        )
        return self._emit_integer_binop(
            block,
            scope=scope,
            loc=loc,
            stem="group_topology_linear_thread_rank",
            fn=operator.add,
            lhs=components["thread_idx", "x"],
            rhs=x_stride,
        )

    @staticmethod
    def _validate_emittable_topology(lowering_plan):
        if lowering_plan is None:
            return None
        topology = lowering_plan.topology
        participation = lowering_plan.participation
        if topology is None or participation is None:
            raise CoopSinglePhaseRewriteError(
                "cooperative provider storage requires group topology and "
                "participation contracts."
            )
        exact_block_dim = participation.exact_block_dim
        if exact_block_dim is None:
            raise CoopSinglePhaseRewriteError(
                "cooperative provider storage requires exact block dimensions."
            )
        block_threads = exact_block_dim[0] * exact_block_dim[1] * exact_block_dim[2]
        if topology.logical_width * topology.instances != block_threads:
            raise CoopSinglePhaseRewriteError(
                "cooperative provider topology does not cover the exact block "
                "dimensions."
            )
        scope = topology.execution_scope
        if scope is SynchronizationScope.BLOCK:
            if (
                topology.instances != 1
                or topology.logical_width != block_threads
                or topology.instance_index != "cta"
                or topology.thread_rank != "linear_thread_rank"
            ):
                raise CoopSinglePhaseRewriteError(
                    "block-scoped cooperative storage requires canonical "
                    "single-CTA ranks."
                )
        elif scope is SynchronizationScope.WARP:
            width = topology.logical_width
            if (
                width < 1
                or width > 32
                or width & (width - 1)
                or 32 % width != 0
                or topology.instance_index != f"linear_thread_rank / {width}"
                or topology.thread_rank != f"linear_thread_rank % {width}"
            ):
                raise CoopSinglePhaseRewriteError(
                    "warp-scoped cooperative storage requires a power-of-two "
                    "logical width dividing 32 and canonical contiguous ranks."
                )
        elif scope is SynchronizationScope.NONE:
            if (
                topology.logical_width != 1
                or topology.instance_index != "linear_thread_rank"
                or topology.thread_rank != "0"
            ):
                raise CoopSinglePhaseRewriteError(
                    "thread-scoped cooperative storage requires canonical "
                    "per-thread ranks."
                )
        else:
            raise CoopSinglePhaseRewriteError(
                "cuda.coop.numba_mlir provider execution scope "
                f"{scope.value!r} has no storage emitter"
            )
        return topology

    def _emit_storage_instance_index(
        self,
        block: ir.Block,
        *,
        lowering_plan,
        scope: ir.Scope,
        loc: ir.Loc,
    ) -> ir.Var:
        topology = self._validate_emittable_topology(lowering_plan)
        if topology is None or topology.execution_scope is SynchronizationScope.BLOCK:
            return self._emit_integer_constant(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_instance_index",
                value=0,
            )
        linear_rank = self._emit_linear_thread_rank(block, scope=scope, loc=loc)
        if topology.execution_scope is SynchronizationScope.NONE:
            return linear_rank
        logical_width = self._emit_integer_constant(
            block,
            scope=scope,
            loc=loc,
            stem="group_topology_logical_width",
            value=topology.logical_width,
        )
        return self._emit_integer_binop(
            block,
            scope=scope,
            loc=loc,
            stem="group_topology_instance_index",
            fn=operator.floordiv,
            lhs=linear_rank,
            rhs=logical_width,
        )

    def _has_temp_storage_requirements(self) -> bool:
        implicit = getattr(self, "_implicit_temp_storage_requirements", None)
        return bool(self._func_temp_storage_requirements) or bool(
            implicit is not None and implicit.uses
        )

    def _get_device_shared_memory_limits(self, required_bytes: int) -> tuple[int, int]:
        conservative_default = _DEFAULT_STATIC_SHARED_MEMORY_BYTES
        if required_bytes <= conservative_default:
            return (conservative_default, conservative_default)
        try:
            limits = _query_device_shared_memory_limits()
            max_default = int(limits["max_default_shared_memory_per_block"])
            max_optin = int(limits["max_optin_shared_memory_per_block"])
        except (
            AttributeError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise CoopSinglePhaseRewriteError(
                "TempStorage requirements above the conservative 49152-byte "
                "static shared-memory limit require an exact current-device "
                "shared-memory query"
            ) from exc
        if max_default <= 0 or max_optin <= 0 or max_optin < max_default:
            raise CoopSinglePhaseRewriteError(
                "The current device reported invalid shared-memory limits: "
                f"default={max_default}, opt-in={max_optin}."
            )
        return (max_default, max_optin)

    def _ensure_temp_storage_global_plan(self) -> _TempStorageGlobalPlan:
        cached = self._temp_storage_global_plan
        if cached is not None:
            return cached
        ordered_keys = sorted(
            {
                self._canonical_temp_storage_ctor_key(key)
                for key in self._func_temp_storage_requirements
            },
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
        implicit = getattr(
            self,
            "_implicit_temp_storage_requirements",
            _TempStorageRequirementSummary(),
        )
        if implicit.uses:
            (
                implicit_size,
                implicit_alignment,
                implicit_slices,
            ) = self._layout_temp_storage_uses(
                implicit.uses,
                sharing="shared",
            )
            offset = _align_up(offset, implicit_alignment)
            implicit_base_offset = offset
            self._implicit_temp_storage_plan = _TempStoragePlan(
                size_in_bytes=implicit_size,
                alignment=implicit_alignment,
                sharing="shared",
                auto_sync=True,
                slices_by_call_id=implicit_slices,
                base_offset=implicit_base_offset,
            )
            offset += implicit_size
            max_alignment = max(max_alignment, implicit_alignment)
        else:
            self._implicit_temp_storage_plan = None
        total_size = _align_up(offset, max_alignment)
        max_default, max_optin = self._get_device_shared_memory_limits(total_size)
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

    def _stage_temp_storage_backing(self) -> ir.Var:
        """Insert the aggregate allocation before rewriting any consumer."""

        if self._temp_storage_backing_emitted:
            if self._temp_storage_backing_var is None:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage backing was marked emitted without an IR value."
                )
            return self._temp_storage_backing_var
        plan = self._ensure_temp_storage_global_plan()
        entry_block = self._func_ir.blocks[min(self._func_ir.blocks)]
        staged = ir.Block(entry_block.scope, entry_block.loc)
        backing = self._emit_temp_storage_backing(staged, plan=plan)
        insert_at = 0
        while insert_at < len(entry_block.body):
            statement = entry_block.body[insert_at]
            if not (
                isinstance(statement, ir.Assign) and isinstance(statement.value, ir.Arg)
            ):
                break
            insert_at += 1
        entry_block.body[insert_at:insert_at] = staged.body
        entry_block.verify()
        return backing

    def _emit_temp_storage_backing(
        self, block: ir.Block, *, plan: _TempStorageGlobalPlan
    ) -> ir.Var:
        if self._temp_storage_backing_emitted:
            if self._temp_storage_backing_var is None:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage backing was marked emitted without an IR value."
                )
            return self._temp_storage_backing_var
        loc = block.loc
        scope = block.scope
        module_var = ir.Var(
            scope,
            f"__coop_temp_storage_module_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        shared_var = ir.Var(
            scope,
            f"__coop_temp_storage_shared_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        array_fn_var = ir.Var(
            scope,
            f"__coop_temp_storage_array_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        bytes_var = ir.Var(
            scope,
            f"__coop_temp_storage_bytes_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        align_var = ir.Var(
            scope,
            f"__coop_temp_storage_alignment_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        dtype_var = ir.Var(
            scope,
            f"__coop_temp_storage_dtype_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        backing_var = ir.Var(
            scope,
            f"__coop_temp_storage_backing_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        block.append(
            ir.Assign(
                ir.Global(
                    _next_global_name("temp_storage_module"),
                    _cuda_module,
                    loc,
                ),
                module_var,
                loc,
            )
        )
        block.append(
            ir.Assign(ir.Expr.getattr(module_var, "shared", loc), shared_var, loc)
        )
        block.append(
            ir.Assign(ir.Expr.getattr(shared_var, "array", loc), array_fn_var, loc)
        )
        alloc_size = 0 if plan.uses_dynamic_smem else int(plan.total_size)
        block.append(ir.Assign(ir.Const(alloc_size, loc), bytes_var, loc))
        block.append(ir.Assign(ir.Const(plan.max_alignment, loc), align_var, loc))
        block.append(
            ir.Assign(
                ir.Global(
                    _next_global_name("temp_storage_dtype"),
                    _cuda_module.uint8,
                    loc,
                ),
                dtype_var,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.call(
                    array_fn_var,
                    [bytes_var, dtype_var],
                    (("alignment", align_var),),
                    loc,
                ),
                backing_var,
                loc,
            )
        )
        self._temp_storage_backing_var = backing_var
        self._temp_storage_backing_emitted = True
        return backing_var

    def _emit_array_slice(
        self,
        block: ir.Block,
        *,
        source_var: ir.Var,
        target_var: ir.Var,
        start: int | ir.Var,
        stop: int | ir.Var,
        loc: ir.Loc,
    ) -> None:
        slice_ctor_global_name = _next_global_name("temp_storage_slice_ctor")
        slice_ctor_var = ir.Var(
            target_var.scope,
            f"__coop_temp_storage_slice_ctor_var_{next(_GLOBAL_NAME_COUNTER)}__",
            loc,
        )
        start_var = start
        if not isinstance(start_var, ir.Var):
            start_var = self._emit_integer_constant(
                block,
                scope=target_var.scope,
                loc=loc,
                stem="temp_storage_slice_start",
                value=start_var,
            )
        stop_var = stop
        if not isinstance(stop_var, ir.Var):
            stop_var = self._emit_integer_constant(
                block,
                scope=target_var.scope,
                loc=loc,
                stem="temp_storage_slice_stop",
                value=stop_var,
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

    def _emit_temp_storage_slice_for_call(
        self,
        block: ir.Block,
        *,
        source_var: ir.Var,
        target_var: ir.Var,
        slice_info: _TempStorageSlice,
        base_offset: int,
        loc: ir.Loc,
    ) -> None:
        static_start = int(base_offset) + int(slice_info.offset)
        if slice_info.instances == 1:
            start: int | ir.Var = static_start
            stop: int | ir.Var = static_start + int(slice_info.size_in_bytes)
        else:
            if slice_info.lowering_plan is None:
                raise CoopSinglePhaseRewriteError(
                    "multi-instance cooperative storage requires a group lowering plan."
                )
            instance_index = self._emit_storage_instance_index(
                block,
                lowering_plan=slice_info.lowering_plan,
                scope=target_var.scope,
                loc=loc,
            )
            stride = self._emit_integer_constant(
                block,
                scope=target_var.scope,
                loc=loc,
                stem="temp_storage_instance_stride",
                value=(
                    int(slice_info.size_in_bytes)
                    if slice_info.stride is None
                    else int(slice_info.stride)
                ),
            )
            instance_offset = self._emit_integer_binop(
                block,
                scope=target_var.scope,
                loc=loc,
                stem="temp_storage_instance_offset",
                fn=operator.mul,
                lhs=instance_index,
                rhs=stride,
            )
            domain_offset = self._emit_integer_constant(
                block,
                scope=target_var.scope,
                loc=loc,
                stem="temp_storage_domain_offset",
                value=static_start,
            )
            start = self._emit_integer_binop(
                block,
                scope=target_var.scope,
                loc=loc,
                stem="temp_storage_slice_start",
                fn=operator.add,
                lhs=domain_offset,
                rhs=instance_offset,
            )
            slice_size = self._emit_integer_constant(
                block,
                scope=target_var.scope,
                loc=loc,
                stem="temp_storage_slice_size",
                value=slice_info.size_in_bytes,
            )
            stop = self._emit_integer_binop(
                block,
                scope=target_var.scope,
                loc=loc,
                stem="temp_storage_slice_stop",
                fn=operator.add,
                lhs=start,
                rhs=slice_size,
            )
        self._emit_array_slice(
            block,
            source_var=source_var,
            target_var=target_var,
            start=start,
            stop=stop,
            loc=loc,
        )

    def _runtime_temp_storage_arg_for_call(
        self, block: ir.Block, *, source_var: ir.Var, call_assign: ir.Assign
    ) -> tuple[ir.Var, _TempStoragePlan | None]:
        temp_storage_arg = source_var
        temp_storage_plan = self._resolve_temp_storage_plan(source_var)
        if temp_storage_plan is not None:
            slice_info = temp_storage_plan.slices_by_call_id.get(id(call_assign))
            if slice_info is None:
                raise CoopSinglePhaseRewriteError(
                    f"Could not resolve TempStorage slice for call at {call_assign.loc}."
                )
            if temp_storage_plan.sharing == "exclusive" or slice_info.offset != 0:
                sliced_var = ir.Var(
                    call_assign.target.scope,
                    f"__coop_temp_storage_slice_{next(_GLOBAL_NAME_COUNTER)}__",
                    call_assign.loc,
                )
                self._emit_temp_storage_slice_for_call(
                    block,
                    source_var=source_var,
                    target_var=sliced_var,
                    slice_info=slice_info,
                    base_offset=0,
                    loc=call_assign.loc,
                )
                temp_storage_arg = sliced_var
        return (temp_storage_arg, temp_storage_plan)

    def _implicit_temp_storage_arg_for_call(
        self, block: ir.Block, *, call_assign: ir.Assign
    ) -> tuple[ir.Var, _TempStoragePlan]:
        plan = self._implicit_temp_storage_plan
        backing = self._temp_storage_backing_var
        if plan is None or backing is None:
            raise CoopSinglePhaseRewriteError(
                "Missing implementation-owned TempStorage plan for an implicit call."
            )
        slice_info = plan.slices_by_call_id.get(id(call_assign))
        if slice_info is None:
            raise CoopSinglePhaseRewriteError(
                f"Could not resolve implicit TempStorage slice for call at {call_assign.loc}."
            )
        sliced_var = ir.Var(
            call_assign.target.scope,
            f"__coop_implicit_temp_storage_slice_{next(_GLOBAL_NAME_COUNTER)}__",
            call_assign.loc,
        )
        self._emit_temp_storage_slice_for_call(
            block,
            source_var=backing,
            target_var=sliced_var,
            slice_info=slice_info,
            base_offset=plan.base_offset,
            loc=call_assign.loc,
        )
        return (sliced_var, plan)

    def _emit_temp_storage_auto_sync(
        self,
        block: ir.Block,
        *,
        scope: ir.Scope,
        loc: ir.Loc,
        synchronization_scope: SynchronizationScope,
        lowering_plan=None,
    ) -> None:
        synchronization_scope = SynchronizationScope(synchronization_scope)
        if synchronization_scope is SynchronizationScope.NONE:
            return
        topology = self._validate_emittable_topology(lowering_plan)
        if topology is not None and (
            synchronization_scope is not topology.execution_scope
        ):
            raise CoopSinglePhaseRewriteError(
                "cooperative provider synchronization scope disagrees with "
                "its group topology."
            )
        sync_attr = {
            SynchronizationScope.WARP: "syncwarp",
            SynchronizationScope.BLOCK: "syncthreads",
        }.get(synchronization_scope)
        if sync_attr is None:
            raise CoopSinglePhaseRewriteError(
                "cuda.coop.numba_mlir provider synchronization scope "
                f"{SynchronizationScope(synchronization_scope).value!r} "
                "has no emitter"
            )
        sync_args = []
        if (
            synchronization_scope is SynchronizationScope.WARP
            and topology is not None
            and topology.logical_width < 32
        ):
            linear_rank = self._emit_linear_thread_rank(
                block,
                scope=scope,
                loc=loc,
            )
            lane_mask = self._emit_integer_constant(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_lane_mask",
                value=31,
            )
            lane = self._emit_integer_binop(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_lane",
                fn=operator.and_,
                lhs=linear_rank,
                rhs=lane_mask,
            )
            logical_width = self._emit_integer_constant(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_logical_width",
                value=topology.logical_width,
            )
            logical_group = self._emit_integer_binop(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_logical_group",
                fn=operator.floordiv,
                lhs=lane,
                rhs=logical_width,
            )
            shift = self._emit_integer_binop(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_mask_shift",
                fn=operator.mul,
                lhs=logical_group,
                rhs=logical_width,
            )
            base_mask = self._emit_integer_constant(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_base_mask",
                value=(1 << topology.logical_width) - 1,
            )
            sync_args.append(
                self._emit_integer_binop(
                    block,
                    scope=scope,
                    loc=loc,
                    stem="group_topology_sync_mask",
                    fn=operator.lshift,
                    lhs=base_mask,
                    rhs=shift,
                )
            )
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
            ir.Assign(
                ir.Expr.call(sync_fn_var, sync_args, (), loc), sync_result_var, loc
            )
        )

    def _temp_storage_alias_ctor_key(self, inst: ir.Assign) -> str | None:
        value = inst.value
        if isinstance(value, ir.Var):
            sources = (value,)
        elif isinstance(value, ir.Expr) and value.op == "cast":
            sources = (getattr(value, "value", None),)
        elif isinstance(value, ir.Expr) and value.op == "phi":
            sources = _phi_incoming_values(value)
        else:
            return None
        if not sources or any(not isinstance(source, ir.Var) for source in sources):
            return None
        keys = [self._resolve_temp_storage_ctor_key(source) for source in sources]
        if any(key is None for key in keys):
            return None
        return self._resolve_temp_storage_ctor_key(inst.target)

    def _validate_temp_storage_uses(
        self, func_ir, matches: dict[ir.Assign, _RewriteMatch]
    ) -> None:
        """Ensure TempStorage descriptors only feed a primitive keyword."""

        if not self._temp_storage_ctor_specs:
            for match in matches.values():
                if match.runtime_temp_storage_var is not None:
                    raise CoopSinglePhaseRewriteError(
                        "cooperative group temp_storage= must originate from a "
                        "TempStorage constructor in the compiled function."
                    )
            return

        consumed_ctor_keys: set[str] = set()
        for label in sorted(func_ir.blocks):
            scan_block = func_ir.blocks[label]
            self._block = scan_block
            self._block_defs = {
                inst.target.name: inst.value
                for inst in scan_block.body
                if isinstance(inst, ir.Assign)
            }
            for inst in scan_block.body:
                used_vars = list(inst.list_vars())
                if isinstance(inst, ir.Assign):
                    used_vars = [
                        var for var in used_vars if var.name != inst.target.name
                    ]
                descriptor_vars = []
                for value in used_vars:
                    if self._resolve_temp_storage_ctor_key(value) is not None:
                        descriptor_vars.append(value)
                if not descriptor_vars:
                    match = matches.get(inst)
                    if match is not None and match.runtime_temp_storage_var is not None:
                        raise CoopSinglePhaseRewriteError(
                            "cooperative group temp_storage= must originate from a "
                            "TempStorage constructor in the compiled function."
                        )
                    continue
                if isinstance(inst, ir.Assign) and (
                    self._temp_storage_alias_ctor_key(inst) is not None
                ):
                    continue
                match = matches.get(inst)
                if match is not None and match.runtime_temp_storage_var is not None:
                    storage_var = match.runtime_temp_storage_var
                    storage_key = self._resolve_temp_storage_ctor_key(storage_var)
                    keyword_storage_vars = [
                        value
                        for name, value in inst.value.kws
                        if name == "temp_storage"
                    ]
                    descriptor_runtime_args = [
                        value
                        for value in match.runtime_args
                        if self._resolve_temp_storage_ctor_key(value) is not None
                    ]
                    if (
                        storage_key is not None
                        and len(keyword_storage_vars) == 1
                        and keyword_storage_vars[0].name == storage_var.name
                        and not descriptor_runtime_args
                        and all(
                            value.name == storage_var.name for value in descriptor_vars
                        )
                    ):
                        consumed_ctor_keys.add(storage_key)
                        continue
                names = ", ".join(sorted({value.name for value in descriptor_vars}))
                raise CoopSinglePhaseRewriteError(
                    "TempStorage values are opaque compile-time descriptors and "
                    "may only be passed as temp_storage= to a registered "
                    "cooperative primitive; use "
                    f"involving {names!r} would escape to runtime."
                )

        constructor_keys = {
            self._canonical_temp_storage_ctor_key(key)
            for key in self._temp_storage_ctor_specs
        }
        consumed_ctor_keys = {
            self._canonical_temp_storage_ctor_key(key) for key in consumed_ctor_keys
        }
        unconsumed = constructor_keys - consumed_ctor_keys
        if unconsumed:
            names = ", ".join(sorted(unconsumed))
            raise CoopSinglePhaseRewriteError(
                "TempStorage values are opaque compile-time descriptors and "
                "must be passed as temp_storage= to a registered cooperative "
                "primitive; constructor(s) "
                f"{names!r} have no primitive consumer."
            )

    def _compute_func_temp_storage_requirements(
        self, func_ir
    ) -> dict[str, _TempStorageRequirementSummary]:
        requirements: dict[str, _TempStorageRequirementSummary] = {}
        saved_block_defs = self._block_defs
        saved_block = self._block
        self._temp_storage_ctor_specs = {}
        self._temp_storage_ctor_order = {}
        self._temp_storage_ctor_roots = {}
        self._implicit_temp_storage_requirements = _TempStorageRequirementSummary()
        self._implicit_temp_storage_plan = None
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
            all_matches: list[_RewriteMatch] = []
            matches_by_assign: dict[ir.Assign, _RewriteMatch] = {}
            storage_uses: list[tuple[int, ir.Assign, _RewriteMatch, str | None]] = []
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
                    lowering_plan = factory_kwargs.pop(_GROUP_LOWERING_PLAN_KWARG, None)
                    family_metadata = self._analyze_family_match(
                        op_name=op_name,
                        runtime_args=runtime_args,
                        factory_kwargs=factory_kwargs,
                    )
                    match = _RewriteMatch(
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
                        lowering_plan=lowering_plan,
                    )
                    all_matches.append(match)
                    matches_by_assign[inst] = match
                    ctor_key = (
                        None
                        if runtime_temp_storage_var is None
                        else self._resolve_temp_storage_ctor_key(
                            runtime_temp_storage_var
                        )
                    )
                    if match.factory_metadata.storage_abi is StorageABI.LEADING_POINTER:
                        self._validate_storage_match_plan(match)
                        storage_uses.append((current_order, inst, match, ctor_key))
            self._validate_temp_storage_uses(func_ir, matches_by_assign)
            self._prepare_ltoir_bundle_for_matches(all_matches)
            for use_order, inst, match, ctor_key in storage_uses:
                if ctor_key is not None:
                    ctor_key = self._canonical_temp_storage_ctor_key(ctor_key)
                invocable, _ = self._materialize_invocable(match)
                size_in_bytes = max(
                    1, int(getattr(invocable, "temp_storage_bytes", 0) or 0)
                )
                alignment = max(
                    1, int(getattr(invocable, "temp_storage_alignment", 0) or 0)
                )
                summary = (
                    self._implicit_temp_storage_requirements
                    if ctor_key is None
                    else requirements.setdefault(
                        ctor_key, _TempStorageRequirementSummary()
                    )
                )
                summary.max_size_in_bytes = max(
                    summary.max_size_in_bytes, size_in_bytes
                )
                summary.max_alignment = max(summary.max_alignment, alignment)
                summary.uses.append(
                    _TempStorageUseRequirement(
                        call_assign=inst,
                        order=use_order,
                        size_in_bytes=size_in_bytes,
                        alignment=alignment,
                        lowering_plan=match.lowering_plan,
                    )
                )
        finally:
            self._block_defs = saved_block_defs
            self._block = saved_block
        return requirements


__all__ = ["_StorageRewrite"]
