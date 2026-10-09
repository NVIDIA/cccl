# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Allocate shared scratch and insert reuse barriers for cooperative calls.

The call rewrite scans every block before replacing descriptors so all
provider scratch requirements contribute to one function-wide allocation.
Explicit ``TempStorage`` descriptors and implementation-owned scratch receive
aligned regions and per-call views, partitioned where separate group instances
require independent storage. Emission places the backing allocation before its
consumers and adds the synchronization required by each plan's reuse contract.
"""

from __future__ import annotations

import operator
from dataclasses import replace
from typing import TYPE_CHECKING, cast

from numba_cuda_mlir import cuda as _cuda_module
from numba_cuda_mlir.extending import set_required_dynamic_shared_memory
from numba_cuda_mlir.numba_cuda.types import uint8

from cuda.coop._core import (
    GroupLoweringPlan,
    StorageOwnership,
    SynchronizationScope,
)

from ._operations import (
    StorageABI,
    expected_storage_reuse_barrier,
    provider_synchronization_matches,
)
from ._rewrite_support import (
    _DEFAULT_STATIC_SHARED_MEMORY_BYTES,
    _DYNAMIC_SHARED_MEMORY_ALIGNMENT,
    _GLOBAL_NAME_COUNTER,
    _INFERENCE_EXCEPTIONS,
    CoopSinglePhaseRewriteError,
    _align_up,
    _next_global_name,
    _normalize_temp_storage_alignment,
    _phi_incoming_values,
    _query_device_shared_memory_limits,
    _RewriteMatch,
    _TempStorageGlobalPlan,
    _TempStoragePlan,
    _TempStorageRequirementSummary,
    _TempStorageSlice,
    _TempStorageUseRequirement,
    ir,
)

if TYPE_CHECKING:
    from cuda.coop._core import GroupTopologyRequirements

    from ._rewrite import CoopSinglePhaseRewrite


class _StorageRewrite:
    """Plan scratch across the function and emit each call's storage view.

    The provenance mixin supplies descriptor owners and region layouts. This
    mixin joins those regions, checks placement limits, and writes allocation,
    slice, and synchronization statements into the IR.
    """

    def _validate_storage_match_plan(
        self, match: _RewriteMatch, *, ctor_key: str | None = None
    ) -> None:
        """Check storage contracts before allocating slices and barriers.

        Validate the provider registry contract against the group planner's
        execution topology, storage ownership, shared address space, and reuse
        barrier. Explicit caller-owned storage is supported only for a single
        block instance. With manual synchronization, that caller-owned case
        may retain the provider's execution-scope synchronization declaration
        even though this rewrite emits no automatic reuse barrier.

        The group planner and this rewrite parse descriptors independently.
        When a constructor key is available, compare their effective contracts
        so a parser disagreement cannot silently suppress synchronization.
        Calls without a lowering plan retain only the legacy block execution
        and block synchronization contract. Requirement collection runs these
        checks before materializing storage-bearing invocables.

        Parameters
        ----------
        match : _RewriteMatch
            Validated call with a leading storage pointer in its provider ABI.
        ctor_key : str or None, optional
            Explicit descriptor owner used to cross-check constructor
            metadata. None applies to implicit storage or when no owner
            was resolved.

        Returns
        -------
        None
            Successful return permits subsequent requirement collection.

        Raises
        ------
        CoopSinglePhaseRewriteError
            The plan cannot be emitted or any provider, ownership,
            synchronization, or constructor contract disagrees.
        """

        rewrite = cast("CoopSinglePhaseRewrite", self)
        cls = type(self)
        lowering_plan = match.lowering_plan
        if lowering_plan is None:
            if (
                match.factory_metadata.execution_scope
                is not SynchronizationScope.BLOCK
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
                "cooperative provider storage received an unsupported group "
                "lowering plan."
            )
        topology = cls._validate_emittable_topology(lowering_plan)
        synchronization = lowering_plan.synchronization
        storage = lowering_plan.temp_storage
        if topology is None or synchronization is None or storage is None:
            raise CoopSinglePhaseRewriteError(
                "cooperative provider storage requires complete group "
                "topology, synchronization, and storage contracts."
            )
        if (
            match.factory_metadata.execution_scope
            is not topology.execution_scope
        ):
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
                "cooperative provider storage layout disagrees with its "
                "group topology."
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
        expected_reuse_barrier = expected_storage_reuse_barrier(
            topology, storage
        )
        planned_reuse_barrier = synchronization.storage_reuse_barrier
        if planned_reuse_barrier is not expected_reuse_barrier:
            raise CoopSinglePhaseRewriteError(
                "cooperative provider TempStorage automatic synchronization "
                "disagrees with its planned storage-reuse barrier."
            )
        if not provider_synchronization_matches(
            match.factory_metadata, topology, synchronization, storage
        ):
            raise CoopSinglePhaseRewriteError(
                "cooperative provider synchronization scope disagrees with "
                "its group lowering plan."
            )
        if caller_owned and ctor_key is not None:
            # The group planner and this rewrite parse the same constructor
            # independently. The barrier is emitted only when both agree, so
            # any drift must fail loudly rather than drop the barrier.
            specification = rewrite._temp_storage_ctor_specifications.get(
                rewrite._canonical_temp_storage_ctor_key(ctor_key)
            )
            if specification is not None:
                size, alignment, auto_sync, sharing = (
                    rewrite._temp_storage_contract(specification)
                )
                planned_alignment = storage.requested_alignment
                if planned_alignment is not None:
                    planned_alignment = _normalize_temp_storage_alignment(
                        planned_alignment
                    )
                planned = (
                    storage.requested_size_in_bytes,
                    planned_alignment,
                    storage.auto_sync,
                    storage.sharing,
                )
                if (size, alignment, auto_sync, sharing) != planned:
                    raise CoopSinglePhaseRewriteError(
                        "cooperative provider TempStorage contract "
                        "disagrees between the group lowering plan "
                        f"{planned!r} and the descriptor "
                        f"{(size, alignment, auto_sync, sharing)!r}."
                    )

    @staticmethod
    def _emit_integer_constant(
        block: ir.Block,
        *,
        scope: ir.Scope | None,
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
        scope: ir.Scope | None,
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
        scope: ir.Scope | None,
        loc: ir.Loc,
    ) -> ir.Var:
        """Append IR for the CUDA thread's linear rank within its block.

        Compute ``threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y *
        threadIdx.z)`` from runtime CUDA indices. Storage instance selection
        and logical-warp barrier masks need this rank rather than
        ``threadIdx.x`` for blocks launched with more than one dimension.

        Parameters
        ----------
        block : ir.Block
            IR block receiving index reads and integer arithmetic.
        scope : ir.Scope or None
            Scope assigned to generated variables.
        loc : ir.Loc
            Source location for generated statements and variables.

        Returns
        -------
        ir.Var
            Variable containing the linear block rank, with the x
            dimension varying fastest.
        """

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
                    _next_global_name("group_topology_module"),
                    _cuda_module,
                    loc,
                ),
                module_var,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.getattr(module_var, "threadIdx", loc),
                thread_idx_var,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.getattr(module_var, "blockDim", loc), block_dim_var, loc
            )
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
                    ir.Assign(
                        ir.Expr.getattr(aggregate, component, loc), value, loc
                    )
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
    def _validate_emittable_topology(
        lowering_plan: GroupLoweringPlan | None,
    ) -> GroupTopologyRequirements | None:
        """Require group rank formulas that the storage emitters support.

        A plan must cover every complete group in the exact block dimensions.
        Accept a single block, contiguous logical groups within each physical
        warp, or individual threads. Check the symbolic instance and rank
        expressions against those forms; emitters implement
        these specific formulas rather than evaluating arbitrary topology
        expression strings.

        Parameters
        ----------
        lowering_plan : GroupLoweringPlan or None
            Plan whose topology and participation requirements govern
            storage. None preserves the legacy block-provider path.

        Returns
        -------
        GroupTopologyRequirements or None
            Validated topology, or None when no plan was supplied.

        Raises
        ------
        CoopSinglePhaseRewriteError
            Dimensions or execution requirements are missing, coverage is
            inconsistent, or no emitter supports the scope and rank formulas.
        """

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
        block_threads = (
            exact_block_dim[0] * exact_block_dim[1] * exact_block_dim[2]
        )
        participating_threads = block_threads
        nonexhaustive_warp = (
            participation.group_kind == "threads_within_warp"
            and not participation.complete_parent_partition
            and 1 <= topology.logical_width <= 32
        )
        if nonexhaustive_warp:
            participating_threads = (
                (block_threads // 32)
                * (32 // topology.logical_width)
                * topology.logical_width
            )
        if topology.logical_width * topology.instances != participating_threads:
            raise CoopSinglePhaseRewriteError(
                "cooperative provider topology does not cover the exact "
                "block dimensions."
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
                or (width & (width - 1) and block_threads % 32 != 0)
            ):
                raise CoopSinglePhaseRewriteError(
                    "warp-scoped cooperative storage requires a logical width "
                    "from 1 through 32; non-power-of-two widths require "
                    "complete physical warps."
                )
            if nonexhaustive_warp:
                instance_index = (
                    f"(linear_thread_rank / 32) * {32 // width} + "
                    f"((linear_thread_rank % 32) / {width})"
                )
                thread_rank = f"(linear_thread_rank % 32) % {width}"
            else:
                instance_index = f"linear_thread_rank / {width}"
                thread_rank = f"linear_thread_rank % {width}"
            if (
                topology.instance_index != instance_index
                or topology.thread_rank != thread_rank
            ):
                raise CoopSinglePhaseRewriteError(
                    "warp-scoped cooperative storage requires canonical "
                    "ranks within each physical warp."
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
        lowering_plan: GroupLoweringPlan | None,
        scope: ir.Scope | None,
        loc: ir.Loc,
    ) -> ir.Var:
        """Select the executing group's region within a scratch allocation.

        Scratch slicing calls this when a block contains several independent
        groups. Each warp or thread needs its own region so concurrent
        operations do not overwrite one another. Use zero for a block-wide
        group, the linear thread rank for a thread, or that rank divided by the
        logical warp width. Non-exhaustive warp groups restart the index
        within each physical warp so trailing lanes consume no scratch.

        Parameters
        ----------
        block : ir.Block
            Replacement block receiving the index calculation.
        lowering_plan : GroupLoweringPlan or None
            Validated operation topology that defines group size and rank.
            ``None`` uses the single-region block convention of private provider
            calls.
        scope : ir.Scope or None
            Scope for the generated index variables.
        loc : ir.Loc
            Original call location used for emitted IR and diagnostics.

        Returns
        -------
        ir.Var
            Zero-based group instance index. This is a region index, not a byte
            offset; the caller multiplies it by the aligned region stride.
        """

        topology = self._validate_emittable_topology(lowering_plan)
        if (
            topology is None
            or topology.execution_scope is SynchronizationScope.BLOCK
        ):
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
        if 32 % topology.logical_width:
            physical_width = self._emit_integer_constant(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_physical_width",
                value=32,
            )
            physical_warp = self._emit_integer_binop(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_physical_warp",
                fn=operator.floordiv,
                lhs=linear_rank,
                rhs=physical_width,
            )
            lane = self._emit_integer_binop(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_lane",
                fn=operator.mod,
                lhs=linear_rank,
                rhs=physical_width,
            )
            groups_per_warp = self._emit_integer_constant(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_groups_per_warp",
                value=32 // topology.logical_width,
            )
            warp_offset = self._emit_integer_binop(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_warp_offset",
                fn=operator.mul,
                lhs=physical_warp,
                rhs=groups_per_warp,
            )
            local_instance = self._emit_integer_binop(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_local_instance",
                fn=operator.floordiv,
                lhs=lane,
                rhs=logical_width,
            )
            return self._emit_integer_binop(
                block,
                scope=scope,
                loc=loc,
                stem="group_topology_instance_index",
                fn=operator.add,
                lhs=warp_offset,
                rhs=local_instance,
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
        rewrite = cast("CoopSinglePhaseRewrite", self)
        implicit = getattr(self, "_implicit_temp_storage_requirements", None)
        return bool(rewrite._func_temp_storage_requirements) or bool(
            implicit is not None and implicit.uses
        )

    def _get_device_shared_memory_limits(
        self, required_bytes: int
    ) -> tuple[int, int]:
        """Get the default and opt-in byte limits needed for this allocation.

        Requests up to the conservative static limit need no device query.
        Larger ``required_bytes`` values require valid limits from the active
        CUDA device; a failed query becomes a rewrite error.
        """

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
        """Plan one shared backing for explicit and implicit scratch.

        Finalize canonical explicit descriptors in constructor order and
        assign aligned base offsets, then append the implementation-owned
        region. Calls within each region have already been assigned reusable
        or exclusive slices. Round the total size to the greatest alignment so
        the backing can satisfy every region with one allocation.

        Use static shared memory when the total fits the default device limit;
        otherwise request the exact dynamic byte count in compiler metadata.
        Small allocations use the conservative limit without querying a
        device. Dynamic placement must fit the opt-in limit and the compiler's
        dynamic window alignment guarantee. Cache the plan and updated region
        offsets. Allocation and checks for user shared arrays happen later.

        Returns
        -------
        _TempStorageGlobalPlan
            Cached or newly computed total size, maximum alignment,
            placement, and required dynamic byte count.

        Raises
        ------
        CoopSinglePhaseRewriteError
            A descriptor cannot be finalized, exact device limits are
            required but unavailable, or size/alignment requirements
            exceed supported limits.
        """

        rewrite = cast("CoopSinglePhaseRewrite", self)
        cached = self._temp_storage_global_plan
        if cached is not None:
            return cached
        ordered_keys = sorted(
            {
                rewrite._canonical_temp_storage_ctor_key(key)
                for key in rewrite._func_temp_storage_requirements
            },
            key=lambda name: (
                rewrite._temp_storage_ctor_order.get(name, 1 << 30),
                name,
            ),
        )
        offset = 0
        max_alignment = 1
        for key in ordered_keys:
            plan = rewrite._finalize_temp_storage_plan_for_var(key)
            alignment = max(1, int(plan.alignment))
            offset = _align_up(offset, alignment)
            rewrite._temp_storage_plans[key] = replace(plan, base_offset=offset)
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
            ) = rewrite._layout_temp_storage_uses(
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
        max_default, max_optin = self._get_device_shared_memory_limits(
            total_size
        )
        uses_dynamic_smem = total_size > max_default
        if (
            uses_dynamic_smem
            and max_alignment > _DYNAMIC_SHARED_MEMORY_ALIGNMENT
        ):
            # The static path honors the requested alignment through the
            # shared array declaration; the dynamic window only guarantees
            # its declared alignment, and telling the optimizer otherwise
            # would be a false assumption.
            raise CoopSinglePhaseRewriteError(
                f"TempStorage requires {max_alignment}-byte alignment, but "
                f"the {total_size}-byte backing exceeds the "
                f"{max_default}-byte static shared-memory limit and dynamic "
                "shared memory guarantees only "
                f"{_DYNAMIC_SHARED_MEMORY_ALIGNMENT}-byte alignment; reduce "
                "the requested alignment or the storage size."
            )
        dynamic_shared_bytes = total_size if uses_dynamic_smem else 0
        if dynamic_shared_bytes > max_optin:
            raise CoopSinglePhaseRewriteError(
                f"TempStorage requires {dynamic_shared_bytes} bytes dynamic "
                f"shared memory, but device max opt-in is {max_optin} bytes."
            )
        if dynamic_shared_bytes > 0:
            set_required_dynamic_shared_memory(
                rewrite._state, dynamic_shared_bytes
            )
        plan = _TempStorageGlobalPlan(
            total_size=total_size,
            max_alignment=max_alignment,
            uses_dynamic_smem=uses_dynamic_smem,
            dynamic_shared_bytes=dynamic_shared_bytes,
        )
        self._temp_storage_global_plan = plan
        return plan

    def _stage_temp_storage_backing(self) -> ir.Var:
        """Stage the scratch allocation before rewriting consumers.

        Place generated allocation statements in the entry block immediately
        after argument assignments. Block rewrite visitation need not follow
        control flow; emitting next to the first visited consumer could leave
        other consumers without a dominating definition. Reject conflicting
        user shared-memory declarations before inserting the unified backing.

        Repeated calls return the existing backing variable without inserting
        a second allocation. The function's entry block is mutated directly,
        even when ``apply`` is currently rewriting a different block.

        Returns
        -------
        ir.Var
            Shared byte array available to every rewritten consumer.

        Raises
        ------
        CoopSinglePhaseRewriteError
            Allocation planning fails, user shared arrays would overlap,
            or the emission state claims a backing exists without
            recording its variable.
        """
        rewrite = cast("CoopSinglePhaseRewrite", self)

        if self._temp_storage_backing_emitted:
            if self._temp_storage_backing_var is None:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage backing was marked emitted "
                    "without an IR value."
                )
            return self._temp_storage_backing_var
        plan = self._ensure_temp_storage_global_plan()
        self._reject_conflicting_user_shared_arrays(plan)
        entry_block = rewrite._func_ir.blocks[min(rewrite._func_ir.blocks)]
        staged = ir.Block(entry_block.scope, entry_block.loc)
        backing = self._emit_temp_storage_backing(staged, plan=plan)
        insert_at = 0
        while insert_at < len(entry_block.body):
            statement = entry_block.body[insert_at]
            if not (
                isinstance(statement, ir.Assign)
                and isinstance(statement.value, ir.Arg)
            ):
                break
            insert_at += 1
        entry_block.body[insert_at:insert_at] = staged.body
        entry_block.verify()
        return backing

    def _reject_conflicting_user_shared_arrays(
        self, plan: _TempStorageGlobalPlan
    ) -> None:
        """Reject shared allocations that overlap in the supported compiler.

        User runtime-sized or zero-sized shared arrays use the dynamic
        window; static allocations currently overlap that window. Coexistence
        is allowed only when both sides are static. Inspect constructor origins
        after inlining and restore lookup state if a diagnostic is raised.
        """
        rewrite = cast("CoopSinglePhaseRewrite", self)
        uses_dynamic_smem = plan.uses_dynamic_smem
        saved_block = self._block
        saved_block_defs = self._block_defs
        conflicts: list[tuple[str, ir.Loc]] = []
        try:
            for label in sorted(rewrite._func_ir.blocks):
                scan_block = rewrite._func_ir.blocks[label]
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
                    if not rewrite._is_shared_array_ctor_call(call):
                        continue
                    shape_ref = (
                        call.args[0]
                        if call.args
                        else dict(call.kws).get("shape")
                    )
                    try:
                        shape = rewrite._infer_constant(shape_ref)
                    except _INFERENCE_EXCEPTIONS:
                        shape = None
                    dimensions = shape if isinstance(shape, tuple) else (shape,)
                    is_static = bool(dimensions) and all(
                        isinstance(extent, int) and extent > 0
                        for extent in dimensions
                    )
                    if uses_dynamic_smem or not is_static:
                        placement = (
                            "static" if is_static else "dynamic/runtime-sized"
                        )
                        conflicts.append((placement, inst.loc))
        finally:
            self._block = saved_block
            self._block_defs = saved_block_defs
        if not conflicts:
            return
        placement = "dynamic" if uses_dynamic_smem else "static"
        requirement = (
            "cuda.coop temporary storage requires a "
            f"{plan.total_size}-byte {placement} shared-memory backing"
        )
        where = ", ".join(
            f"{kind} cuda.shared.array(...) at {loc}"
            for kind, loc in conflicts[:3]
        )
        raise CoopSinglePhaseRewriteError(
            f"{requirement}, but "
            f"this kernel also declares {where}. The supported numba-cuda-mlir "
            "compiler does not separate these allocations; they would alias. "
            "Use statically sized user shared arrays and keep the combined "
            "cooperative backing within the static shared-memory limit, or "
            "move the user data out of shared memory."
        )

    def _emit_temp_storage_backing(
        self, block: ir.Block, *, plan: _TempStorageGlobalPlan
    ) -> ir.Var:
        """Append one shared byte-array allocation and remember its variable.

        ``plan`` supplies capacity, alignment, and static or dynamic
        placement. Dynamic placement declares a zero-length shared array,
        which the compiler maps to the launch's dynamic shared memory.
        Planning has already recorded the required byte count. Repeated
        calls return the existing variable without appending IR.
        """

        if self._temp_storage_backing_emitted:
            if self._temp_storage_backing_var is None:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage backing was marked emitted "
                    "without an IR value."
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
            ir.Assign(
                ir.Expr.getattr(module_var, "shared", loc), shared_var, loc
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.getattr(shared_var, "array", loc), array_fn_var, loc
            )
        )
        alloc_size = 0 if plan.uses_dynamic_smem else int(plan.total_size)
        block.append(ir.Assign(ir.Const(alloc_size, loc), bytes_var, loc))
        block.append(
            ir.Assign(ir.Const(plan.max_alignment, loc), align_var, loc)
        )
        block.append(
            ir.Assign(
                ir.Global(
                    _next_global_name("temp_storage_dtype"),
                    uint8,
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
        """Assign a view of a scratch allocation to a provider operand.

        Storage-view emission calls this after computing the required bounds.
        The generated slice shares the backing array, allowing each provider to
        receive its assigned region without allocating or copying scratch data.
        Append constants as needed, the slice object, and the view assignment.

        Parameters
        ----------
        block : ir.Block
            Replacement block to which the slice statements are appended.
        source_var : ir.Var
            Backing array or descriptor view to slice.
        target_var : ir.Var
            Variable receiving the view; its scope is used for new temporaries.
        start, stop : int or ir.Var
            Inclusive start and exclusive stop in array elements. Scratch arrays
            contain bytes, so their element indices are also byte offsets.
            Bounds may be constants or variables computed for the current group.
        loc : ir.Loc
            Source location attached to generated statements.
        """

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
                ir.Global(slice_ctor_global_name, slice, loc),
                slice_ctor_var,
                loc,
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
            ir.Assign(
                ir.Expr.getitem(source_var, slice_obj_var, loc), target_var, loc
            )
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
        """Emit a call's scratch view for the executing group.

        A single-instance view has constant bounds. Multiple instances add the
        emitted group index times the aligned instance stride to the region's
        static offset. The view length remains the call's byte requirement,
        even if the stride includes padding or room for another consumer.

        Parameters
        ----------
        block : ir.Block
            IR block receiving bound calculations and the slice assignment.
        source_var : ir.Var
            Shared byte array or descriptor view to slice.
        target_var : ir.Var
            Variable to receive the view and supply its IR scope.
        slice_info : _TempStorageSlice
            Offset in the region, call size, instance layout, and topology.
        base_offset : int
            Region origin relative to ``source_var``. Use zero for an
            already sliced explicit descriptor and the region base for the
            unified backing.
        loc : ir.Loc
            Source location for generated statements.

        Returns
        -------
        None
            ``target_var`` is defined by statements appended to ``block``.

        Raises
        ------
        CoopSinglePhaseRewriteError
            A multi-instance slice lacks a lowering plan or has
            unsupported topology.
        """

        static_start = int(base_offset) + int(slice_info.offset)
        if slice_info.instances == 1:
            start: int | ir.Var = static_start
            stop: int | ir.Var = static_start + int(slice_info.size_in_bytes)
        else:
            if slice_info.lowering_plan is None:
                raise CoopSinglePhaseRewriteError(
                    "multi-instance cooperative storage requires a group "
                    "lowering plan."
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
        """Choose the explicit scratch view passed to one provider call.

        ``apply`` uses the function-wide storage plan to replace a descriptor
        operand with the call's assigned region. The provider receives its own
        byte extent even when a reservation or another primitive requires a
        larger shared region.

        Parameters
        ----------
        block : ir.Block
            Replacement block receiving a slice assignment when one is needed.
        source_var : ir.Var
            Current storage operand, used to find its descriptor's plan.
        call_assign : ir.Assign
            Original provider-call assignment. Its object identity selects the
            recorded slice, and its target supplies the scope for emitted
            variables.

        Returns
        -------
        tuple of ir.Var and _TempStoragePlan or None
            Provider storage operand and its plan for subsequent
            synchronization. If no descriptor plan exists, return ``(source_var,
            None)`` unchanged.
        """

        rewrite = cast("CoopSinglePhaseRewrite", self)
        temp_storage_arg = source_var
        temp_storage_plan = rewrite._resolve_temp_storage_plan(source_var)
        if temp_storage_plan is not None:
            slice_info = temp_storage_plan.slices_by_call_id.get(
                id(call_assign)
            )
            if slice_info is None:
                raise CoopSinglePhaseRewriteError(
                    "Could not resolve TempStorage slice for call at "
                    f"{call_assign.loc}."
                )
            if (
                temp_storage_plan.sharing == "exclusive"
                or slice_info.offset != 0
                or slice_info.size_in_bytes != temp_storage_plan.size_in_bytes
            ):
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
        """Build a scratch operand for a call that omitted a storage descriptor.

        ``apply`` calls this after allocating the function's shared backing
        array. The implementation-owned region has already been sized across
        implicit consumers; this method selects the current call's part and
        returns its plan for the later synchronization step.

        Parameters
        ----------
        block : ir.Block
            Replacement block receiving the call's scratch-view statements.
        call_assign : ir.Assign
            Original provider-call assignment. Its object identity selects the
            recorded slice, and its target supplies the scope for emitted
            variables.

        Returns
        -------
        tuple of ir.Var and _TempStoragePlan
            Generated scratch operand and the implicit region's plan.

        Raises
        ------
        CoopSinglePhaseRewriteError
            The backing array, implicit region plan, or call's slice is missing.
        """

        plan = self._implicit_temp_storage_plan
        backing = self._temp_storage_backing_var
        if plan is None or backing is None:
            raise CoopSinglePhaseRewriteError(
                "Missing implementation-owned TempStorage plan for an "
                "implicit call."
            )
        slice_info = plan.slices_by_call_id.get(id(call_assign))
        if slice_info is None:
            raise CoopSinglePhaseRewriteError(
                "Could not resolve implicit TempStorage slice for call at "
                f"{call_assign.loc}."
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
        scope: ir.Scope | None,
        loc: ir.Loc,
        synchronization_scope: SynchronizationScope,
        lowering_plan: GroupLoweringPlan | None = None,
    ) -> None:
        """Emit the post-call barrier required for automatic scratch reuse.

        Emit ``syncthreads`` for block scope and ``syncwarp`` for warp scope;
        ``NONE`` emits nothing. A logical warp smaller than 32 receives a mask
        covering only its contiguous lanes within the physical warp, computed
        from the linear block rank. Full warps use the default warp mask.

        This is the post-call storage-reuse barrier selected by the lowering
        contract. It neither establishes uniform participation nor decides
        whether a call needs synchronization; the caller has already checked
        the storage policy and calls this emitter for automatic sync.

        Parameters
        ----------
        block : ir.Block
            Destination receiving mask calculations and the barrier call.
        scope : ir.Scope or None
            Scope assigned to generated variables.
        loc : ir.Loc
            Source location for generated statements and variables.
        synchronization_scope : SynchronizationScope
            Requested reuse-barrier scope, converted to the enum on entry.
        lowering_plan : GroupLoweringPlan or None, optional
            Group topology for validation and logical-warp mask
            construction. None uses the legacy scope-only emission path.

        Returns
        -------
        None
            Append barrier IR in place unless the requested scope is ``NONE``.

        Raises
        ------
        CoopSinglePhaseRewriteError
            The topology is unsupported or conflicts with the requested scope.
        """

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
                f"{SynchronizationScope(synchronization_scope).value!r} has "
                "no emitter"
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
                ir.Expr.getattr(sync_module_var, sync_attr, loc),
                sync_fn_var,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.call(sync_fn_var, sync_args, (), loc),
                sync_result_var,
                loc,
            )
        )

    def _temp_storage_alias_ctor_key(self, inst: ir.Assign) -> str | None:
        """Recognize an assignment that only forwards a storage descriptor.

        Accept copies, casts, and merged variables only when every source has
        known constructor provenance. Return the target's canonical owner so
        descriptor-use validation can allow the alias.
        """

        rewrite = cast("CoopSinglePhaseRewrite", self)
        value = inst.value
        if isinstance(value, ir.Var):
            sources = (value,)
        elif isinstance(value, ir.Expr) and value.op == "cast":
            sources = (getattr(value, "value", None),)
        elif isinstance(value, ir.Expr) and value.op == "phi":
            sources = _phi_incoming_values(value)
        else:
            return None
        if not sources or any(
            not isinstance(source, ir.Var) for source in sources
        ):
            return None
        keys = [
            rewrite._resolve_temp_storage_ctor_key(cast(ir.Var, source))
            for source in sources
        ]
        if any(key is None for key in keys):
            return None
        return rewrite._resolve_temp_storage_ctor_key(inst.target)

    def _validate_temp_storage_uses(
        self, func_ir: ir.FunctionIR, matches: dict[ir.Assign, _RewriteMatch]
    ) -> None:
        """Reject storage descriptors that escape their compile-time role.

        After recording all constructors and matches, inspect every use
        through its alias provenance. Allow simple alias, cast, and phi
        assignments and one ``temp_storage=`` keyword on a recognized
        primitive; descriptor values cannot be ordinary runtime operands,
        returned objects, or arbitrary call arguments. Subscripted-provider
        syntax is not an accepted descriptor use. Recognized reserve methods
        are validated separately. Every canonical constructor must have a
        primitive or reservation consumer.

        Validation runs after device-helper inlining. If a call that passes a
        descriptor to a helper remains, report that the helper was not
        inlined. Perform these checks before compiling providers so invalid
        descriptor escapes fail without materialization. The scan updates the
        current block lookup state; its caller restores it.

        Parameters
        ----------
        func_ir : FunctionIR
            Entire function whose constructors have been recorded.
        matches : dict of ir.Assign to _RewriteMatch
            Recognized calls keyed by their original IR assignments.

        Returns
        -------
        None
            All descriptor uses and constructor consumers are valid.

        Raises
        ------
        CoopSinglePhaseRewriteError
            A storage argument lacks local constructor provenance, a
            descriptor escapes to runtime, or a constructor has no
            primitive consumer.
        """
        rewrite = cast("CoopSinglePhaseRewrite", self)

        if not rewrite._temp_storage_ctor_specifications:
            for match in matches.values():
                if match.runtime_temp_storage_var is not None:
                    raise CoopSinglePhaseRewriteError(
                        "cooperative group temp_storage= must originate from "
                        "a TempStorage constructor in the compiled function."
                    )
            return

        consumed_ctor_keys = {
            rewrite._canonical_temp_storage_ctor_key(reservation.ctor_key)
            for reservation in rewrite._temp_storage_reservations.values()
        }
        for label in sorted(func_ir.blocks):
            scan_block = func_ir.blocks[label]
            self._block = scan_block
            self._block_defs = {
                inst.target.name: inst.value
                for inst in scan_block.body
                if isinstance(inst, ir.Assign)
            }
            for inst in scan_block.body:
                if inst in rewrite._temp_storage_reserve_methods:
                    continue
                used_vars = list(inst.list_vars())
                if isinstance(inst, ir.Assign):
                    used_vars = [
                        var for var in used_vars if var.name != inst.target.name
                    ]
                descriptor_vars = []
                for value in used_vars:
                    if (
                        rewrite._resolve_temp_storage_ctor_key(value)
                        is not None
                    ):
                        descriptor_vars.append(value)
                if not descriptor_vars:
                    match = matches.get(inst)
                    if (
                        match is not None
                        and match.runtime_temp_storage_var is not None
                    ):
                        raise CoopSinglePhaseRewriteError(
                            "cooperative group temp_storage= must originate "
                            "from a TempStorage constructor "
                            "in the compiled function."
                        )
                    continue
                if isinstance(inst, ir.Assign) and (
                    self._temp_storage_alias_ctor_key(inst) is not None
                ):
                    continue
                match = matches.get(inst)
                if (
                    match is not None
                    and match.runtime_temp_storage_var is not None
                ):
                    storage_var = match.runtime_temp_storage_var
                    storage_key = rewrite._resolve_temp_storage_ctor_key(
                        storage_var
                    )
                    assert isinstance(inst.value, ir.Expr)
                    keyword_storage_vars = [
                        value
                        for name, value in inst.value.kws
                        if name == "temp_storage"
                    ]
                    descriptor_runtime_args = [
                        value
                        for value in match.runtime_args
                        if rewrite._resolve_temp_storage_ctor_key(value)
                        is not None
                    ]
                    if (
                        storage_key is not None
                        and len(keyword_storage_vars) == 1
                        and keyword_storage_vars[0].name == storage_var.name
                        and not descriptor_runtime_args
                        and all(
                            value.name == storage_var.name
                            for value in descriptor_vars
                        )
                    ):
                        consumed_ctor_keys.add(storage_key)
                        continue
                names = ", ".join(
                    sorted({value.name for value in descriptor_vars})
                )
                if (
                    isinstance(inst, ir.Assign)
                    and isinstance(inst.value, ir.Expr)
                    and inst.value.op == "call"
                    and (
                        helper := rewrite._resolve_python_value(inst.value.func)
                    )
                    is not None
                    and rewrite._is_jitted_dispatcher(helper)
                ):
                    helper_name = helper.py_func.__qualname__
                    raise CoopSinglePhaseRewriteError(
                        f"TempStorage descriptor {names!r} is passed to a "
                        "device function that was not inlined into this "
                        f"kernel ({helper_name!r}); let Numba-CUDA-MLIR "
                        "inline the primitive helper (inline='always') or "
                        "move its cooperative calls into the kernel."
                    )
                raise CoopSinglePhaseRewriteError(
                    "TempStorage values are opaque compile-time descriptors "
                    "and may only be passed as temp_storage= to a "
                    "registered cooperative primitive or used with reserve(); "
                    "a use involving "
                    f"{names!r} would escape to runtime."
                )

        constructor_keys = {
            rewrite._canonical_temp_storage_ctor_key(key)
            for key in rewrite._temp_storage_ctor_specifications
        }
        consumed_ctor_keys = {
            rewrite._canonical_temp_storage_ctor_key(key)
            for key in consumed_ctor_keys
        }
        unconsumed = constructor_keys - consumed_ctor_keys
        if unconsumed:
            names = ", ".join(sorted(unconsumed))
            raise CoopSinglePhaseRewriteError(
                "TempStorage values are opaque compile-time descriptors and "
                "must be passed as temp_storage= to a registered "
                "cooperative primitive or used with reserve(); "
                f"constructor(s) {names!r} have no consumer."
            )

    def _collect_temp_storage_uses(
        self, func_ir: ir.FunctionIR, matches: dict[ir.Assign, _RewriteMatch]
    ) -> list[tuple[int, ir.Assign, _RewriteMatch, str | None]]:
        """Validate scratch ownership before provider compilation.

        Consume the analyzed calls in source order, retaining each storage
        use's assignment and descriptor owner. Check all descriptor escapes
        before returning, including calls whose ABI has no scratch pointer.
        """

        rewrite = cast("CoopSinglePhaseRewrite", self)
        storage_uses: list[
            tuple[int, ir.Assign, _RewriteMatch, str | None]
        ] = []
        saved_block_defs = self._block_defs
        saved_block = self._block
        try:
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
                    match = matches.get(inst)
                    if match is None:
                        continue
                    ctor_key = (
                        None
                        if match.runtime_temp_storage_var is None
                        else rewrite._resolve_temp_storage_ctor_key(
                            match.runtime_temp_storage_var
                        )
                    )
                    if (
                        match.factory_metadata.storage_abi
                        is StorageABI.LEADING_POINTER
                    ):
                        self._validate_storage_match_plan(
                            match, ctor_key=ctor_key
                        )
                        storage_uses.append(
                            (current_order, inst, match, ctor_key)
                        )
            self._validate_temp_storage_uses(func_ir, matches)
        finally:
            self._block_defs = saved_block_defs
            self._block = saved_block
        return storage_uses

    def _compute_func_temp_storage_requirements(
        self,
        storage_uses: list[tuple[int, ir.Assign, _RewriteMatch, str | None]],
    ) -> dict[str, _TempStorageRequirementSummary]:
        """Collect sizes and alignments from validated provider specializations.

        Bundle preparation has already supplied layouts where possible.
        Materialization reuses those artifacts or compiles individual
        providers as a fallback. Explicit descriptors accumulate under their
        canonical owners; implicit scratch has a separate summary. Allocation
        offsets are assigned later, when emitting the shared backing.
        """

        rewrite = cast("CoopSinglePhaseRewrite", self)
        requirements: dict[str, _TempStorageRequirementSummary] = {}
        for use_order, inst, match, ctor_key in storage_uses:
            if ctor_key is not None:
                ctor_key = rewrite._canonical_temp_storage_ctor_key(ctor_key)
            invocable, _ = rewrite._materialize_invocable(match)
            size_in_bytes = max(
                1, int(getattr(invocable, "temp_storage_bytes", 0) or 0)
            )
            alignment = max(
                1, int(getattr(invocable, "temp_storage_alignment", 0) or 0)
            )
            summary = (
                rewrite._implicit_temp_storage_requirements
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
        rewrite._add_temp_storage_reservations(requirements)
        return requirements


__all__ = ["_StorageRewrite"]
