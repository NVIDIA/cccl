# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lower planned primitives and per-thread payloads to executable Numba IR.

``CoopWholeFunctionPlanner`` runs after device helpers have been inlined and
before type inference. Its group-resolution phase replaces public primitive
calls with private provider calls. This module specializes those providers
into callable implementations (invocables), supplies their scratch storage,
and replaces ``ThreadData`` constructors with per-thread local arrays.

``_CallRewriting._rewrite_calls`` drives the block-level ``match``/``apply``
helpers. It also handles ``ThreadData`` used in ordinary indexed computation,
without a cooperative primitive. ``TempStorage`` is opaque scratch and must
be passed to a registered primitive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from numba_cuda_mlir import cuda as _cuda_module
from numba_cuda_mlir.extending import require_launch_config

from cuda.coop._core import GroupLoweringPlan

from ._operations import _GROUP_LOWERING_PLAN_KWARG, StorageABI
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
    _DeferredCoopRewrite,
    _next_global_name,
    _RewriteMatch,
    ir,
)

if TYPE_CHECKING:
    from numba_cuda_mlir.numba_cuda.types import Type
    from numba_cuda_mlir.numba_cuda.typing.templates import Signature

    from ._planner import CoopWholeFunctionPlanner
    from ._rewrite_support import _ThreadDataSpecification


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
    """Replace one block's provider calls using a plan for the whole function.

    The mixins share payload type/extent information, scratch-storage plans,
    and provider caches. This lets argument inference and array allocation
    use the same information across calls and control-flow branches.

    ``match`` gathers a block's work; ``apply`` performs it. Both use the
    function-wide payload and scratch plan so calls in different blocks use
    compatible allocations. Only the whole-function planner invokes them;
    the mixins are not independently scheduled compiler passes.
    """

    def match(
        self,
        func_ir: ir.FunctionIR,
        block: ir.Block,
        typemap: dict[str, Type] | None,
        calltypes: dict[ir.Expr, Signature] | None,
    ) -> bool:
        """Collect the work for one block before type inference.

        The whole-function planner calls this after group resolution. First
        collect payload and scratch requirements across all blocks. Then
        record this block's constructors, extent queries, and provider calls
        for ``apply``.

        The function-wide scan can compile providers and update compiler
        caches. It leaves block statements unchanged. Missing launch facts
        defer the work; ``_rewrite_calls`` requests those facts and rescans
        with a fresh rewriter. The rewrite-interface maps ``typemap`` and
        ``calltypes`` are not read here.
        """

        from ._group_planner import has_group_markers

        if has_group_markers(func_ir):
            return False
        if not self._prepare_function(func_ir):
            return False
        self._prepare_block(block)
        for inst in block.body:
            if isinstance(inst, ir.Assign):
                self._match_assignment(inst)
        if self._deferred_launch_dim_inference:
            # Preserve this block for a fresh scan once launch dimensions are
            # available. _rewrite_calls requests them within this compiler
            # attempt; device helpers leave this work for the inlined caller.
            return False

        return (
            bool(self._matches)
            or bool(self._temp_storage_assigns)
            or bool(self._thread_data_func_vars)
            or bool(self._thread_data_extents)
        )

    def _prepare_function(self, func_ir: ir.FunctionIR) -> bool:
        """Collect all storage requirements before rewriting any block.

        A new IR object starts a new storage plan. Keep that plan across
        this function's blocks so constructors include requirements from
        every consumer.
        """

        func_ir_identity = id(func_ir)
        if self._func_ir_identity == func_ir_identity:
            return True
        self._func_ir_identity = func_ir_identity
        self._func_ir = func_ir
        self._thread_data_specifications = {}
        self._thread_data_like_vars = set()
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

        return True

    def _prepare_block(self, block: ir.Block) -> None:
        """Reset block matches and keep the function-wide storage plan."""

        self._block = block
        self._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        self._matches = {}
        self._thread_data_extents = {}
        self._temp_storage_assigns = set()
        self._temp_storage_func_vars = set()
        self._thread_data_func_vars = set()

    def _match_assignment(self, inst: ir.Assign) -> None:
        """Record a payload query, constructor, or provider call."""

        call = inst.value
        if (
            isinstance(call, ir.Expr)
            and call.op == "getattr"
            and call.attr == "items_per_thread"
        ):
            if self._is_thread_data_like_var(call.value):
                specification = self._resolve_thread_data_specification(
                    call.value
                )
                if (
                    specification is not None
                    and specification.items_per_thread is not None
                ):
                    self._thread_data_extents[inst] = (
                        specification.items_per_thread
                    )
            return
        if not isinstance(call, ir.Expr) or call.op != "call":
            return
        if self._is_temp_storage_ctor_call(call):
            self._temp_storage_assigns.add(inst)
            self._temp_storage_func_vars.add(call.func.name)
            self._record_temp_storage_ctor(inst, call)
            self._temp_storage_ctor_order.setdefault(
                inst.target.name, len(self._temp_storage_ctor_order)
            )
            return
        if self._is_thread_data_ctor_call(call):
            self._thread_data_func_vars.add(call.func.name)
            self._thread_data_like_vars.add(inst.target.name)
            self._thread_data_specifications[inst.target.name] = (
                self._merge_thread_data_specifications(
                    self._thread_data_specifications.get(inst.target.name),
                    self._extract_thread_data_specification(call),
                )
            )
            return

        self._match_provider_call(inst, call)

    def _match_provider_call(self, inst: ir.Assign, call: ir.Expr) -> None:
        """Separate compile-time factory inputs from device operands.

        Keep the group lowering plan beside the match. It controls storage
        and synchronization during emission, but is not an argument to the
        factory.
        """

        target = self._resolve_call_target(call)
        if target is None:
            return
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
            return
        lowering_plan = cast(
            GroupLoweringPlan | None,
            factory_kwargs.pop(_GROUP_LOWERING_PLAN_KWARG, None),
        )
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
            lowering_plan=lowering_plan,
        )

    def apply(self) -> ir.Block:
        """Emit the most recently matched block before type inference.

        Stage shared backing storage, bind provider implementations, then
        replace each statement in source order. Payloads become local
        arrays. Scratch descriptors become shared-memory views or ``None``
        for storage-free calls. Provider calls receive their device operands
        and any required reuse barrier.

        Finally remove unused compile-time bindings and refresh the typing
        context. Backing storage and payload-binding cleanup can also change
        other blocks. The caller installs the returned block and repairs
        function IR analysis.
        """

        assert self._block is not None
        # Stage shared backing storage before emitting any scratch view.
        if self._has_temp_storage_requirements():
            self._stage_temp_storage_backing()
        (
            call_invocable_globals,
            func_var_names_to_clear,
            candidate_dead_factory_kw_vars,
        ) = self._prepare_call_invocables()

        new_block = ir.Block(self._block.scope, self._block.loc)
        for inst in self._block.body:
            if inst in self._thread_data_extents:
                new_block.append(
                    ir.Assign(
                        ir.Const(self._thread_data_extents[inst], inst.loc),
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
                and (self._is_thread_data_ctor_call(inst.value))
            ):
                self._emit_thread_data_array(new_block, inst)
                continue
            match = self._matches.get(inst)
            if match is None and inst not in self._temp_storage_assigns:
                new_block.append(inst)
                continue
            if inst in self._temp_storage_assigns:
                self._emit_temp_storage(new_block, inst)
                continue
            assert match is not None
            self._emit_provider_call(
                new_block, inst, match, call_invocable_globals.get(inst)
            )

        # Other blocks can still use factory inputs and constructor aliases.
        new_block = self._remove_unused_factory_arguments(
            new_block, candidate_dead_factory_kw_vars
        )
        self._clear_unused_payload_callees(new_block)
        self._state.typingctx.refresh()
        return new_block

    def _prepare_call_invocables(
        self,
    ) -> tuple[dict[ir.Assign, tuple[str, object]], set[str], set[str]]:
        """Prepare each call's implementation and track bindings to retire.

        The function-wide scan has already attempted batch compilation.
        Reuse those invocables where available; individual materialization
        is the fallback.
        """

        call_invocable_globals: dict[ir.Assign, tuple[str, object]] = {}
        func_var_names_to_clear: set[str] = set()
        candidate_dead_factory_kw_vars: set[str] = set()
        # Build or reuse each provider implementation before emitting its call.
        # Bind it per call site: two calls through the same Python alias can
        # require different specializations.
        for match_inst, match in self._matches.items():
            invocable, _ = self._materialize_invocable(match)
            self._record_invocable_specialization(invocable)
            candidate_dead_factory_kw_vars.update(
                value_var.name for value_var in match.factory_kw_value_vars
            )
            global_name = _next_global_name("single_phase")
            call_invocable_globals[match_inst] = (global_name, invocable)
            func_var_names_to_clear.add(match.func_var_name)
            if match.func_var_name_extra is not None:
                func_var_names_to_clear.add(match.func_var_name_extra)
        return (
            call_invocable_globals,
            func_var_names_to_clear,
            candidate_dead_factory_kw_vars,
        )

    def _require_thread_data_specification(
        self, inst: ir.Assign
    ) -> _ThreadDataSpecification:
        """Resolve the payload dtype before emitting a local array.

        Primitive operands can supply the dtype during requirement
        collection. If they did not, inspect typed indexed writes before
        reporting an error.
        """

        thread_data_specification = self._thread_data_specifications.get(
            inst.target.name
        )
        if (
            thread_data_specification is not None
            and thread_data_specification.dtype is None
        ):
            self._infer_thread_data_dtype_from_writes(inst.target)
            thread_data_specification = self._thread_data_specifications.get(
                inst.target.name
            )
        if (
            thread_data_specification is None
            or thread_data_specification.dtype is None
        ):
            raise CoopSinglePhaseRewriteError(
                "Failed to infer dtype for coop.ThreadData(...). "
                "Supply type information through a cooperative "
                "operation, typed indexed assignments, or an "
                "explicit dtype argument."
            )
        if thread_data_specification.common_root:
            from ._parameters import _validate_common_numeric_dtype

            try:
                _validate_common_numeric_dtype(
                    thread_data_specification.dtype,
                    operation="ThreadData",
                )
            except (TypeError, ValueError) as exc:
                raise CoopSinglePhaseRewriteError(str(exc)) from exc

        return thread_data_specification

    def _thread_data_array_arguments(
        self, new_block: ir.Block, inst: ir.Assign
    ) -> tuple[list[ir.Var], list[tuple[str, ir.Var]]]:
        """Emit constant array parameters and adapt constructor arguments.

        ``cuda.local.array`` takes ``shape`` where ``ThreadData`` takes
        ``items_per_thread``. Preserve the resolved dtype and explicit
        alignment.
        """

        thread_data_specification = self._require_thread_data_specification(
            inst
        )
        dtype_var = ir.Var(
            inst.target.scope,
            f"__coop_thread_data_dtype_{next(_GLOBAL_NAME_COUNTER)}__",
            inst.loc,
        )
        new_block.append(
            ir.Assign(
                ir.Global(
                    _next_global_name("thread_data_dtype"),
                    thread_data_specification.dtype,
                    inst.loc,
                ),
                dtype_var,
                inst.loc,
            )
        )
        assert isinstance(inst.value, ir.Expr)
        rewritten_args = list(inst.value.args)
        rewritten_kws = list(inst.value.kws)
        rewritten_kws = [
            ("shape" if name == "items_per_thread" else name, value)
            for name, value in rewritten_kws
            if name != "alignment"
        ]
        if thread_data_specification.items_per_thread is None:
            raise CoopSinglePhaseRewriteError(
                "Failed to infer items_per_thread for "
                "coop.ThreadData(...). The item count must be "
                "known at compile time."
            )
        items_var = ir.Var(
            inst.target.scope,
            f"__coop_thread_data_items_{next(_GLOBAL_NAME_COUNTER)}__",
            inst.loc,
        )
        new_block.append(
            ir.Assign(
                ir.Const(thread_data_specification.items_per_thread, inst.loc),
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
        if thread_data_specification.alignment is not None:
            alignment_var = ir.Var(
                inst.target.scope,
                f"__coop_thread_data_alignment_{next(_GLOBAL_NAME_COUNTER)}__",
                inst.loc,
            )
            new_block.append(
                ir.Assign(
                    ir.Const(thread_data_specification.alignment, inst.loc),
                    alignment_var,
                    inst.loc,
                )
            )
            rewritten_kws.append(("alignment", alignment_var))

        return rewritten_args, rewritten_kws

    def _emit_thread_data_array(
        self, new_block: ir.Block, inst: ir.Assign
    ) -> None:
        """Replace a payload constructor with ``cuda.local.array``."""

        rewritten_args, rewritten_kws = self._thread_data_array_arguments(
            new_block, inst
        )
        # Bind cuda.local.array at this call site. Other blocks may
        # still need the original ThreadData constructor alias, and
        # a later scan must recognize this call as already lowered.
        array_fn_var = ir.Var(
            inst.target.scope,
            f"__coop_thread_data_array_{next(_GLOBAL_NAME_COUNTER)}__",
            inst.loc,
        )
        module_var = ir.Var(
            inst.target.scope,
            f"__coop_thread_data_module_{next(_GLOBAL_NAME_COUNTER)}__",
            inst.loc,
        )
        local_var = ir.Var(
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
                local_var,
                inst.loc,
            )
        )
        new_block.append(
            ir.Assign(
                ir.Expr.getattr(local_var, "array", inst.loc),
                array_fn_var,
                inst.loc,
            )
        )
        new_block.append(
            ir.Assign(
                ir.Expr.call(
                    array_fn_var,
                    rewritten_args,
                    tuple(rewritten_kws),
                    inst.loc,
                ),
                inst.target,
                inst.loc,
            )
        )

    def _emit_temp_storage(self, new_block: ir.Block, inst: ir.Assign) -> None:
        """Replace a scratch descriptor with its shared-memory view.

        A descriptor whose consumers need no scratch becomes ``None``. Other
        descriptors select their region from the function's shared backing
        array.
        """

        ctor_key = self._resolve_temp_storage_ctor_key(inst.target)
        if ctor_key is None:
            raise CoopSinglePhaseRewriteError(
                f"Missing TempStorage metadata for '{inst.target.name}'."
            )
        if ctor_key not in self._func_temp_storage_requirements:
            # Validation already required a primitive consumer. Its
            # provider needs no scratch, so no array is needed here.
            new_block.append(
                ir.Assign(ir.Const(None, inst.loc), inst.target, inst.loc)
            )
            return
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

    def _emit_provider_call(
        self,
        new_block: ir.Block,
        inst: ir.Assign,
        match: _RewriteMatch,
        call_invocable: tuple[str, object] | None,
    ) -> None:
        """Emit a provider call with its operands and reuse barrier.

        The provider ABI decides whether to prepend a scratch pointer. The
        group and storage plans must agree on automatic synchronization
        before emission of the trailing barrier.
        """

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
        assert isinstance(inst.value, ir.Expr)
        call_func = inst.value.func
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
            and match.lowering_plan is not None
            and match.lowering_plan.temp_storage is not None
            and (
                match.lowering_plan.temp_storage.auto_sync
                is not runtime_temp_storage_plan.auto_sync
            )
        ):
            # The group lowering plan and runtime storage plan must agree
            # on auto_sync before we choose the storage-reuse barrier.
            raise CoopSinglePhaseRewriteError(
                "cooperative provider TempStorage automatic "
                "synchronization disagrees between the group lowering "
                "plan and the descriptor."
            )
        if (
            runtime_temp_storage_plan is not None
            and runtime_temp_storage_plan.auto_sync
        ):
            synchronization_scope = match.factory_metadata.synchronization_scope
            if match.lowering_plan is not None:
                planned_synchronization = match.lowering_plan.synchronization
                if planned_synchronization is None:
                    raise CoopSinglePhaseRewriteError(
                        "cooperative provider storage requires a "
                        "synchronization contract."
                    )
                synchronization_scope = (
                    planned_synchronization.storage_reuse_barrier
                )
            self._emit_temp_storage_auto_sync(
                new_block,
                scope=inst.target.scope,
                loc=inst.loc,
                synchronization_scope=synchronization_scope,
                lowering_plan=match.lowering_plan,
            )

    def _remove_unused_factory_arguments(
        self, new_block: ir.Block, candidate_dead_factory_kw_vars: set[str]
    ) -> ir.Block:
        """Remove factory inputs only after checking uses in every block."""

        used_var_names: set[str] = set()
        # A compile-time argument may still feed a call in another block.
        # Remove its assignment only after checking all remaining uses.
        for block in self._func_ir.blocks.values():
            rewritten_block = new_block if block is self._block else block
            for stmt in rewritten_block.body:
                used_var_names.update(
                    var.name
                    for var in stmt.list_vars()
                    if not isinstance(stmt, ir.Assign)
                    or var.name != stmt.target.name
                )
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

        return new_block

    def _clear_unused_payload_callees(self, new_block: ir.Block) -> None:
        """Clear references to the ThreadData constructor after lowering calls.

        For example, ``constructor = coop.ThreadData`` must remain while a
        call in another block still uses it. Once all calls through that
        binding have been replaced with ``cuda.local.array``, replace the
        binding with ``None`` so type inference need not type the marker
        function. Follow alias chains until no more bindings can be cleared.

        Inspect the whole function with ``new_block`` substituted for the
        current block. This clears constructor function references; the
        payload arrays and computations using them remain in the IR.

        Parameters
        ----------
        new_block : ir.Block
            Replacement block from ``apply``. This block and other function
            blocks may be mutated in place; the candidate set is consumed as
            bindings are retired.

        Returns
        -------
        None
            Constructor function bindings are updated in place.
        """

        blocks = [
            new_block if block is self._block else block
            for block in self._func_ir.blocks.values()
        ]
        candidates = self._thread_data_func_vars
        while candidates:
            used_names = set()
            for block in blocks:
                for stmt in block.body:
                    used_names.update(
                        var.name
                        for var in stmt.list_vars()
                        if not isinstance(stmt, ir.Assign)
                        or var.name != stmt.target.name
                    )
            retired = False
            for block in blocks:
                for stmt in block.body:
                    if not isinstance(stmt, ir.Assign) or (
                        stmt.target.name not in candidates
                        or stmt.target.name in used_names
                    ):
                        continue
                    candidates.remove(stmt.target.name)
                    if isinstance(stmt.value, ir.Var):
                        candidates.add(stmt.value.name)
                    stmt.value = ir.Const(None, stmt.loc)
                    retired = True
            if not retired:
                break


class _CallRewriting:
    """Drive provider-call and payload rewriting across the function."""

    def _rewrite_calls(self) -> bool:
        """Lower providers and payloads, requesting launch dimensions as needed.

        ``CoopWholeFunctionPlanner.run`` invokes this after its
        group-resolution step, even when that step made no changes: a kernel
        can use ``ThreadData`` without any group operations. Visit
        blocks in label order and apply each block's collected matches once per
        scan. The rewrite object sees the inlined consumers when collecting
        payload and storage requirements.

        Public group resolution normally obtains launch facts before this
        method runs. If a remaining private provider call needs dimensions,
        request the configured launch and rescan with a fresh rewrite object,
        this time reporting unresolved dimensions as errors. This is a local
        rescan within the same compiler attempt. ``require_launch_config``
        makes the facts available synchronously; it does not restart Numba
        compilation. Literal-argument requests can separately cause the
        dispatcher to start another compiler attempt.

        A standalone device function has no kernel launch of its own. Leave
        its deferred private calls for rewriting after inlining into a kernel.

        Returns
        -------
        bool
            Whether a replacement block was installed in ``state.func_ir``.

        Raises
        ------
        CoopSinglePhaseRewriteError
            Provider arguments, payloads, or storage violate the rewrite
            requirements, or dimensions remain unresolved after the rescan.
        RuntimeError
            A provider needs the configured launch dimensions, but the
            runtime has no launch configuration or tracker for this attempt.
        """

        planner = cast("CoopWholeFunctionPlanner", self)
        rewrite = CoopSinglePhaseRewrite(planner.state)
        modified = False

        def apply_matches() -> None:
            """Match and rewrite each block once per scan."""

            nonlocal modified
            for label in sorted(planner.state.func_ir.blocks):
                block = planner.state.func_ir.blocks[label]
                if rewrite.match(
                    planner.state.func_ir,
                    block,
                    planner.state.typemap,
                    planner.state.calltypes,
                ):
                    planner.state.func_ir.blocks[label] = rewrite.apply()
                    modified = True

        apply_matches()
        if (
            rewrite._deferred_launch_dim_inference
            and not planner.is_device_function
        ):
            require_launch_config(planner.state)
            # Rebuild inference and storage plans using the now-visible facts.
            rewrite = CoopSinglePhaseRewrite(
                planner.state,
                allow_launch_dim_deferral=False,
            )
            apply_matches()
        return modified


__all__ = [
    "CoopSinglePhaseRewrite",
    "CoopSinglePhaseRewriteError",
    "_CallRewriting",
]
