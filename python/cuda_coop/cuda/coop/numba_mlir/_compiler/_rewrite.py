# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Lower planned primitives and per-thread payloads to executable Numba IR.

``CoopWholeFunctionPlanner`` runs after device helpers have been inlined and
before type inference. Its group-resolution phase replaces public primitive
calls with private provider calls. This module specializes those providers
into callable implementations (invocables), supplies their scratch storage,
and replaces ``ThreadData`` constructors with per-thread local arrays.

``_CallRewriting._rewrite_calls`` owns the function's rewrite lifecycle:
prepare original calls and storage requirements, stage shared backing, rewrite
blocks, then retire compile-time bindings. Preparation must precede every
block replacement: a consumer in a later block can determine the dtype of an
earlier ``ThreadData`` allocation or the requirements of a scratch descriptor.
Block emission still materializes providers as needed and infers standalone
payload dtypes from their writes.

``ThreadData`` is a per-thread payload; it can also be used in ordinary indexed
computation without a primitive. ``TempStorage`` is an opaque descriptor for
cooperative scratch, with ownership and synchronization rules. Its concrete
layout comes from provider preparation. Keeping these records separate lets
payload allocation and scratch views share facts without confusing their
different lifetimes or ownership.
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
from ._rewrite_reserve import _ReservationRewrite
from ._rewrite_storage import _StorageRewrite
from ._rewrite_support import (
    _GLOBAL_NAME_COUNTER,
    CoopSinglePhaseRewriteError,
    Rewrite,
    _DeferredCoopRewrite,
    _next_global_name,
    _RewriteMatch,
    _TempStorageRequirementSummary,
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
    _ReservationRewrite,
    Rewrite,
):
    """Prepare and rewrite cooperative calls across one function.

    The mixins share payload type/extent information, scratch-storage plans,
    and provider caches. This lets argument inference and array allocation
    use the same information across calls and control-flow branches.

    The driver calls ``prepare_calls_and_storage`` before any IR changes and
    ``begin_rewrite`` to stage shared backing. It then uses ``match``/``apply``
    for each block and calls ``finish_rewrite`` after installing all results.
    Original call identity and definitions remain available during emission;
    cleanup waits until no unreplaced consumer can need a compile-time binding.
    These are steps of one planner, not independently scheduled passes.
    """

    def match(
        self,
        func_ir: ir.FunctionIR,
        block: ir.Block,
        typemap: dict[str, Type] | None,
        calltypes: dict[ir.Expr, Signature] | None,
    ) -> bool:
        """Select one block's work using the prepared function's call records.

        ``prepare_calls_and_storage`` must succeed before any block is matched
        or replaced. Provider calls are looked up by original assignment
        identity; constructors and extent queries are collected for ``apply``.
        The rewrite-interface maps ``typemap`` and ``calltypes`` are not read.
        """

        if self._func_ir_identity != id(func_ir):
            raise RuntimeError(
                "prepare_calls_and_storage must succeed before match"
            )
        self._prepare_block(block)
        for inst in block.body:
            if isinstance(inst, ir.Assign):
                self._match_assignment(inst)
        return (
            bool(self._matches)
            or bool(self._temp_storage_assigns)
            or bool(self._thread_data_func_vars)
            or bool(self._typed_group_payload_func_vars)
            or bool(self._thread_data_extents)
            or any(
                inst in self._temp_storage_reservations
                or inst in self._temp_storage_reserve_methods
                for inst in block.body
            )
        )

    def prepare_calls_and_storage(self, func_ir: ir.FunctionIR) -> bool:
        """Prepare all calls and storage across the current function's blocks.

        Device helpers have already been inlined into this one function IR.
        A ``ThreadData`` dtype can come from a consumer in another block, and
        a ``TempStorage`` descriptor's requirements depend on all its consumers.
        Collect their constructors and analyze calls while every original
        definition is available. Validate storage ownership before preparing
        providers, whose concrete layouts supply scratch sizes and alignments.

        Publish completed call records only on success. Missing launch facts
        leave the IR unchanged; the driver either retries with a fresh
        rewriter or leaves a device helper for its inlined caller. Unresolved
        group markers also leave the function for group planning.
        """

        from ._group_planner import has_group_markers

        if has_group_markers(func_ir):
            return False

        func_ir_identity = id(func_ir)
        if self._func_ir_identity == func_ir_identity:
            return True
        self._func_ir_identity = None
        self._func_matches = {}
        self._rewrite_started = False
        self._factory_argument_cleanup_candidates = set()
        self._payload_callee_cleanup_names = set()
        self._func_temp_storage_requirements = {}
        self._implicit_temp_storage_requirements = (
            _TempStorageRequirementSummary()
        )
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
            matches = self._collect_function_calls(func_ir)
            # Invalid ownership or escaping descriptors must fail before
            # provider compilation. The retained calls serve both stages.
            self._collect_temp_storage_reservations(func_ir)
            storage_uses = self._collect_temp_storage_uses(func_ir, matches)
            self._prepare_ltoir_bundle_for_matches(list(matches.values()))
            # Sizes and alignments require concrete provider layouts; the
            # semantic lowering plans alone do not supply them.
            self._func_temp_storage_requirements = (
                self._compute_func_temp_storage_requirements(storage_uses)
            )
        except _DeferredCoopRewrite:
            self._func_temp_storage_requirements = {}
            return False

        self._func_matches = matches
        self._func_ir_identity = func_ir_identity
        return True

    def begin_rewrite(self) -> None:
        """Stage function-wide backing storage before visiting any block.

        Provider layouts are now available. Check shared-memory conflicts
        against the original function, then insert the backing in the entry
        block so it dominates every scratch view, regardless of visit order.
        Preparation and any launch retry must finish before this mutates IR.
        Later block matching ignores the generated backing statements, and
        block rewriting copies them through with the surrounding statements.
        """

        if self._func_ir_identity != id(self._func_ir):
            raise RuntimeError(
                "prepare_calls_and_storage must succeed before rewriting"
            )
        if self._rewrite_started:
            raise RuntimeError("function rewriting has already started")
        if self._has_temp_storage_requirements():
            self._stage_temp_storage_backing()
        self._rewrite_started = True

    def finish_rewrite(self) -> None:
        """Retire compile-time bindings after every replacement is installed.

        A factory input or constructor alias can still feed a call in a later
        block. Wait until all calls have been replaced before checking uses
        across the function. Keep the original definition table throughout
        rewriting: standalone payload inference and constructor-alias cleanup
        both need original definitions. The planner registry repairs IR
        analysis after this lifecycle completes.
        """

        if not self._rewrite_started:
            raise RuntimeError("begin_rewrite must precede finish_rewrite")
        self._remove_unused_factory_arguments()
        self._clear_unused_payload_callees()
        self._factory_argument_cleanup_candidates.clear()
        self._payload_callee_cleanup_names.clear()
        self._rewrite_started = False

    def _collect_function_calls(
        self, func_ir: ir.FunctionIR
    ) -> dict[ir.Assign, _RewriteMatch]:
        """Analyze every provider call before any block is replaced.

        Record all payload and scratch constructors first so consumers in
        other blocks see the same facts. Retain completed call analysis by
        original assignment identity for compilation, storage planning and
        block emission. Replaced assignments no longer match on later visits.
        """

        matches: dict[ir.Assign, _RewriteMatch] = {}
        saved_block_defs = self._block_defs
        saved_block = self._block
        self._temp_storage_ctor_specifications = {}
        self._temp_storage_ctor_order = {}
        self._temp_storage_ctor_roots = {}
        self._temp_storage_ctor_sites = {}
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
                        self._thread_data_like_vars.add(inst.target.name)
                        self._thread_data_specifications[inst.target.name] = (
                            self._merge_thread_data_specifications(
                                self._thread_data_specifications.get(
                                    inst.target.name
                                ),
                                self._extract_thread_data_specification(call),
                            )
                        )
                    elif self._is_typed_group_payload_ctor_call(call):
                        self._thread_data_like_vars.add(inst.target.name)
                        self._thread_data_specifications[inst.target.name] = (
                            self._merge_thread_data_specifications(
                                self._thread_data_specifications.get(
                                    inst.target.name
                                ),
                                self._extract_typed_group_payload_specification(
                                    call
                                ),
                            )
                        )
                    elif self._is_temp_storage_ctor_call(call):
                        self._record_temp_storage_ctor(inst, call)
                        self._temp_storage_ctor_order.setdefault(
                            inst.target.name, ctor_order
                        )
                        ctor_order += 1
            self._validate_temp_storage_ctor_sites()
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
                    match = self._analyze_provider_call(inst, call)
                    if match is not None:
                        matches[inst] = match
        finally:
            self._block_defs = saved_block_defs
            self._block = saved_block
        return matches

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
        self._typed_group_payload_func_vars = set()

    def _match_assignment(self, inst: ir.Assign) -> None:
        """Record a payload query, constructor, or provider call."""

        match = self._func_matches.get(inst)
        if match is not None:
            self._matches[inst] = match
            return
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
            return
        if self._is_thread_data_ctor_call(call):
            self._thread_data_func_vars.add(call.func.name)
            return

        if self._is_typed_group_payload_ctor_call(call):
            self._typed_group_payload_func_vars.add(call.func.name)
            return

    def _analyze_provider_call(
        self, inst: ir.Assign, call: ir.Expr
    ) -> _RewriteMatch | None:
        """Separate compile-time factory inputs from device operands.

        Keep the group lowering plan beside the match. It controls storage
        and synchronization during emission, but is not an argument to the
        factory.
        """

        target = self._resolve_call_target(call)
        if target is None:
            return None
        op_name = target.operation
        # factory_kwargs drives specialization; factory_kw_value_vars tracks
        # original IR bindings for cleanup after all calls are rewritten.
        (
            runtime_args,
            runtime_temp_storage_var,
            factory_kwargs,
            factory_kw_value_vars,
        ) = self._validate_and_split_args(
            op_name, call, target.getitem_temp_storage
        )
        lowering_plan = cast(
            GroupLoweringPlan | None,
            factory_kwargs.pop(_GROUP_LOWERING_PLAN_KWARG, None),
        )
        family_metadata = self._analyze_family_match(
            op_name=op_name,
            runtime_args=runtime_args,
            factory_kwargs=factory_kwargs,
        )
        return _RewriteMatch(
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

        Bind provider implementations, then replace statements in source
        order. ``begin_rewrite`` has staged shared backing. Payloads become
        local arrays. Scratch descriptors become shared-memory views or
        ``None`` for storage-free calls. Provider calls receive device operands
        and any required reuse barrier.

        Retain cleanup candidates for ``finish_rewrite`` so other blocks can
        still use their bindings. Provider materialization and a standalone
        payload's final dtype inference can occur here. The caller installs
        every returned block before finishing the function rewrite.
        """

        assert self._block is not None
        if not self._rewrite_started:
            raise RuntimeError("begin_rewrite must precede apply")
        (
            call_invocable_globals,
            func_var_names_to_clear,
            candidate_dead_factory_kw_vars,
        ) = self._prepare_call_invocables()

        new_block = ir.Block(self._block.scope, self._block.loc)
        for inst in self._block.body:
            if inst in self._temp_storage_reservations:
                self._emit_temp_storage_reservation(new_block, inst)
                continue
            if inst in self._temp_storage_reserve_methods:
                new_block.append(
                    ir.Assign(ir.Const(None, inst.loc), inst.target, inst.loc)
                )
                continue
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
                and (
                    self._is_thread_data_ctor_call(inst.value)
                    or self._is_typed_group_payload_ctor_call(inst.value)
                )
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

        # Only these assignments were eligible for this block's factory-input
        # cleanup. Saving their identity avoids deleting a different binding
        # of the same name after pre-SSA rebinding.
        self._factory_argument_cleanup_candidates.update(
            inst
            for inst in new_block.body
            if isinstance(inst, ir.Assign)
            and inst.target.name in candidate_dead_factory_kw_vars
        )
        self._payload_callee_cleanup_names.update(
            self._thread_data_func_vars | self._typed_group_payload_func_vars
        )
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
        self, inst: ir.Assign, *, is_typed_group_payload: bool
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
            subject = (
                "typed group payload"
                if is_typed_group_payload
                else "coop.ThreadData(...)"
            )
            raise CoopSinglePhaseRewriteError(
                f"Failed to infer dtype for {subject}. Use it with a "
                "cooperative group operation "
                "that provides dtype context."
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
        self,
        new_block: ir.Block,
        inst: ir.Assign,
        *,
        is_typed_group_payload: bool,
    ) -> tuple[list[ir.Var], list[tuple[str, ir.Var]]]:
        """Emit constant array parameters and adapt constructor arguments.

        ``cuda.local.array`` takes ``shape`` where ``ThreadData`` takes
        ``items_per_thread``. Preserve the resolved dtype and explicit
        alignment.
        """

        thread_data_specification = self._require_thread_data_specification(
            inst, is_typed_group_payload=is_typed_group_payload
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
        rewritten_args = [] if is_typed_group_payload else list(inst.value.args)
        rewritten_kws = [] if is_typed_group_payload else list(inst.value.kws)
        rewritten_kws = [
            ("shape" if name == "items_per_thread" else name, value)
            for name, value in rewritten_kws
            if name != "alignment"
        ]
        if thread_data_specification.items_per_thread is None:
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

        assert isinstance(inst.value, ir.Expr)
        is_typed_group_payload = self._is_typed_group_payload_ctor_call(
            inst.value
        )
        rewritten_args, rewritten_kws = self._thread_data_array_arguments(
            new_block, inst, is_typed_group_payload=is_typed_group_payload
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

    def _remove_unused_factory_arguments(self) -> None:
        """Remove eligible factory bindings that no rewritten block uses."""

        if not self._factory_argument_cleanup_candidates:
            return
        used_var_names: set[str] = set()
        for block in self._func_ir.blocks.values():
            for stmt in block.body:
                used_var_names.update(
                    var.name
                    for var in stmt.list_vars()
                    if not isinstance(stmt, ir.Assign)
                    or var.name != stmt.target.name
                )
        for block in self._func_ir.blocks.values():
            block.body = [
                stmt
                for stmt in block.body
                if not (
                    isinstance(stmt, ir.Assign)
                    and stmt in self._factory_argument_cleanup_candidates
                    and stmt.target.name not in used_var_names
                )
            ]

    def _clear_unused_payload_callees(self) -> None:
        """Clear references to payload constructors after lowering their calls.

        For example, ``constructor = coop.ThreadData`` must remain while a
        call in another block still uses it. Once all calls through that
        binding have been replaced with ``cuda.local.array``, replace the
        binding with ``None`` so type inference need not type the marker
        function. Follow alias chains until no more bindings can be cleared.

        Only clear a surviving, unambiguous original definition. A generated
        assignment or another pre-SSA binding with the same name is not a
        constructor alias merely because its name is a cleanup candidate.
        Payload arrays and computations using them remain in the IR.
        """

        if not self._payload_callee_cleanup_names:
            return
        blocks = list(self._func_ir.blocks.values())
        original_definitions = getattr(self._func_ir, "_definitions", {})
        bindings: dict[str, ir.Assign] = {}
        for block in blocks:
            for stmt in block.body:
                if not isinstance(stmt, ir.Assign):
                    continue
                definitions = original_definitions.get(stmt.target.name, ())
                if len(definitions) == 1 and definitions[0] is stmt.value:
                    bindings[stmt.target.name] = stmt
        candidates = set(self._payload_callee_cleanup_names)
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
            for name in tuple(candidates):
                stmt = bindings.get(name)
                if stmt is None or name in used_names:
                    continue
                candidates.remove(name)
                if isinstance(stmt.value, ir.Var):
                    candidates.add(stmt.value.name)
                stmt.value = ir.Const(None, stmt.loc)
                retired = True
            if not retired:
                break


class _CallRewriting:
    """Drive provider-call and payload rewriting across the function."""

    def _rewrite_calls(self) -> bool:
        """Prepare, rewrite, and finish one function, retrying launch facts.

        ``CoopWholeFunctionPlanner.run`` invokes this after its
        group-resolution step, even when that step made no changes: a kernel
        can use ``ThreadData`` without any group operations. Preparation sees
        all original inlined consumers before any block is replaced. Once it
        succeeds, stage backing storage, rewrite blocks in label order, and
        clean up bindings only after installing every replacement. Block
        matching never triggers preparation or handles launch deferral.

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
            """Rewrite a prepared function; leave a deferred one intact."""

            nonlocal modified
            if not rewrite.prepare_calls_and_storage(planner.state.func_ir):
                return
            rewrite.begin_rewrite()
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
            rewrite.finish_rewrite()

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
