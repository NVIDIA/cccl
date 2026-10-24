# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Hand a descriptor's existing region to user code after its final use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from numba_cuda_mlir.numba_cuda.core.errors import NumbaNotImplementedError
from numba_cuda_mlir.numba_cuda.np.numpy_support import as_dtype
from numba_cuda_mlir.numba_cuda.types import uint8

from ._descriptor_provenance import descriptor_definitions
from ._parameters import normalize_dtype_param
from ._rewrite_support import (
    _INFERENCE_EXCEPTIONS,
    CoopSinglePhaseRewriteError,
    _next_global_name,
    _TempStorageRequirementSummary,
    ir,
)

if TYPE_CHECKING:
    from numba_cuda_mlir.numba_cuda.types import Type

    from ._rewrite import CoopSinglePhaseRewrite
    from ._rewrite_support import _RewriteMatch


@dataclass(frozen=True)
class _TempStorageRelease:
    ctor_key: str
    dtype: Type
    itemsize: int
    alignment: int


class _ReleaseRewrite:
    def _release_method_receivers(self, value: ir.Var) -> list[ir.Var]:
        rewrite = cast("CoopSinglePhaseRewrite", self)
        receivers = []
        leaves = list(
            descriptor_definitions(value, rewrite._lookup_definitions)
        )
        for _, leaf in leaves:
            if not (
                isinstance(leaf, ir.Expr)
                and leaf.op == "getattr"
                and leaf.attr == "release"
                and rewrite._resolve_temp_storage_ctor_key(leaf.value)
                is not None
            ):
                return []
            receivers.append(leaf.value)
        return receivers

    def _collect_temp_storage_releases(self, func_ir: ir.FunctionIR) -> None:
        rewrite = cast("CoopSinglePhaseRewrite", self)
        rewrite._temp_storage_releases = {}
        rewrite._temp_storage_release_methods = set()
        saved_block, saved_defs = rewrite._block, rewrite._block_defs
        try:
            for block in func_ir.blocks.values():
                rewrite._block = block
                rewrite._block_defs = {
                    inst.target.name: inst.value
                    for inst in block.body
                    if isinstance(inst, ir.Assign)
                }
                for inst in block.body:
                    if not isinstance(inst, ir.Assign):
                        continue
                    if self._release_method_receivers(inst.target):
                        rewrite._temp_storage_release_methods.add(inst)
                    call = inst.value
                    if not isinstance(call, ir.Expr) or call.op != "call":
                        continue
                    receivers = self._release_method_receivers(call.func)
                    if not receivers:
                        continue
                    owners = {
                        cast(
                            str,
                            rewrite._resolve_temp_storage_ctor_key(receiver),
                        )
                        for receiver in receivers
                    }
                    owner = rewrite._merge_temp_storage_ctor_keys(owners)
                    rewrite._temp_storage_releases[inst] = (
                        self._parse_temp_storage_release(call, owner)
                    )
        finally:
            rewrite._block, rewrite._block_defs = saved_block, saved_defs

        # Bound methods can be copied and called, but cannot escape to a
        # runtime container or an uninlined function.
        methods = {
            inst.target.name for inst in rewrite._temp_storage_release_methods
        }
        for block in func_ir.blocks.values():
            for inst in block.body:
                used = {var.name for var in inst.list_vars()}
                if isinstance(inst, ir.Assign):
                    used.discard(inst.target.name)
                if not used & methods:
                    continue
                if inst in rewrite._temp_storage_release_methods:
                    continue
                if (
                    inst in rewrite._temp_storage_releases
                    and used & methods == {cast(ir.Expr, inst.value).func.name}
                ):
                    continue
                raise CoopSinglePhaseRewriteError(
                    "TempStorage.release bound methods may only be copied "
                    f"or called directly ({inst.loc})."
                )

    def _parse_temp_storage_release(
        self, call: ir.Expr, owner: str
    ) -> _TempStorageRelease:
        rewrite = cast("CoopSinglePhaseRewrite", self)
        if (
            call.vararg is not None
            or getattr(call, "varkwarg", None) is not None
        ):
            raise CoopSinglePhaseRewriteError(
                "TempStorage.release does not accept *args or **kwargs "
                f"({call.loc})."
            )
        if len(call.args) > 1:
            raise CoopSinglePhaseRewriteError(
                f"TempStorage.release accepts only dtype ({call.loc})."
            )
        refs = dict(zip(("dtype",), call.args))
        for name, value in call.kws:
            if name != "dtype" or name in refs:
                raise CoopSinglePhaseRewriteError(
                    f"TempStorage.release got an unexpected or repeated "
                    f"argument {name!r} ({call.loc})."
                )
            refs[name] = value
        try:
            raw = rewrite._infer_constant(refs["dtype"]) if refs else None
            dtype = uint8 if raw is None else normalize_dtype_param(raw)
            numpy_dtype = as_dtype(dtype)
            if numpy_dtype.kind not in "iufc":
                raise ValueError("dtype must be a numeric scalar type")
        except (*_INFERENCE_EXCEPTIONS, NumbaNotImplementedError) as exc:
            raise CoopSinglePhaseRewriteError(
                "TempStorage.release dtype must be a supported compile-time "
                f"numeric scalar type ({call.loc})."
            ) from exc
        return _TempStorageRelease(
            owner,
            dtype,
            numpy_dtype.itemsize,
            max(numpy_dtype.itemsize, numpy_dtype.alignment),
        )

    def _validate_temp_storage_lifetimes(
        self,
        func_ir: ir.FunctionIR,
        matches: dict[ir.Assign, _RewriteMatch],
        *,
        additional_uses: dict[ir.Assign, ir.Var] | None = None,
    ) -> None:
        """Propagate consumption along CFG edges, including loop backedges.

        Track original constructor sites, not the layout planner's merged
        owners. Each variable holds possible (site, release site) pairs.
        Releasing consumes aliases too. A new construction overwrites only its
        target: stale aliases and bound methods stay consumed across iterations.
        """
        rewrite = cast("CoopSinglePhaseRewrite", self)
        if not rewrite._temp_storage_releases:
            return
        blocks = func_ir.blocks
        entry = min(blocks)
        predecessors = {label: [] for label in blocks}
        for label, block in blocks.items():
            for target in block.terminator.get_targets():
                predecessors[target].append(label)
        constructors = {
            inst: id(inst)
            for block in blocks.values()
            for inst in block.body
            if isinstance(inst, ir.Assign)
            and isinstance(inst.value, ir.Expr)
            and inst.value.op == "call"
            and rewrite._is_temp_storage_ctor_call(inst.value)
        }
        outgoing = {}
        event_origins = {}
        unbound = (-1, 0)
        release_locations = {
            id(inst): inst.loc for inst in rewrite._temp_storage_releases
        }
        changed = True
        while changed:
            changed = False
            for label, block in blocks.items():
                parents = [
                    outgoing[p] for p in predecessors[label] if p in outgoing
                ]
                if label != entry and not parents:
                    continue
                names = set().union(*(parent.keys() for parent in parents))
                state = {
                    name: set().union(
                        *(parent.get(name, {unbound}) for parent in parents)
                    )
                    for name in names
                }
                for inst in block.body:
                    release = inst in rewrite._temp_storage_releases
                    match = matches.get(inst)
                    operand = (
                        inst.value.func
                        if release
                        else match.runtime_temp_storage_var
                        if match is not None
                        else None
                    )
                    if operand is None and additional_uses is not None:
                        operand = additional_uses.get(inst)
                    if operand is not None:
                        origins = state.get(operand.name, {unbound})
                        event_origins[inst] = origins
                        if unbound in origins:
                            raise CoopSinglePhaseRewriteError(
                                "TempStorage descriptor or bound method must "
                                "be created on every reachable path before use "
                                f"({inst.loc})."
                            )
                        if any(consumed for _, consumed in origins):
                            action = (
                                "release()"
                                if release
                                else "a cooperative primitive"
                            )
                            released_at = min(
                                str(release_locations[consumed])
                                for _, consumed in origins
                                if consumed
                            )
                            raise CoopSinglePhaseRewriteError(
                                f"TempStorage cannot be used by {action} after "
                                "release() on a reachable path: "
                                f"released at {released_at}; "
                                f"used at {inst.loc}."
                            )
                        if release:
                            sites = {site for site, _ in origins}
                            state = {
                                name: {
                                    (
                                        site,
                                        consumed or id(inst)
                                        if site in sites
                                        else consumed,
                                    )
                                    for site, consumed in values
                                }
                                for name, values in state.items()
                            }
                    if not isinstance(inst, ir.Assign):
                        continue
                    value = inst.value
                    if inst in constructors:
                        origins = {(constructors[inst], 0)}
                    elif isinstance(value, ir.Var):
                        origins = set(state.get(value.name, ()))
                    elif isinstance(value, ir.Expr) and value.op in {
                        "cast",
                        "getattr",
                    }:
                        origins = (
                            set(state.get(value.value.name, ()))
                            if value.op == "cast"
                            or value.attr in {"release", "reserve"}
                            else set()
                        )
                    elif isinstance(value, ir.Expr) and value.op == "phi":
                        origins = set().union(
                            *(
                                outgoing[parent].get(v.name, {unbound})
                                for v, parent in zip(
                                    value.incoming_values,
                                    value.incoming_blocks,
                                )
                                if parent in outgoing
                            )
                        )
                    else:
                        origins = set()
                    if origins:
                        state[inst.target.name] = origins
                    else:
                        state.pop(inst.target.name, None)
                if outgoing.get(label) != state:
                    outgoing[label] = state
                    changed = True
        for inst, origins in event_origins.items():
            if not origins:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage use has no local descriptor instance "
                    f"on a reachable path ({inst.loc})."
                )

    def _validate_released_region_owners(self) -> None:
        """Keep released native views independent after descriptor joins.

        The existing scratch planner can merge compatible descriptors for
        automatic reuse. A native view escapes that contract, so reject such
        joins instead of returning overlapping arrays for independent sites.
        """
        rewrite = cast("CoopSinglePhaseRewrite", self)
        for inst, release in rewrite._temp_storage_releases.items():
            owner = rewrite._canonical_temp_storage_ctor_key(release.ctor_key)
            sites = set().union(
                *(
                    sites
                    for key, sites in rewrite._temp_storage_ctor_sites.items()
                    if rewrite._canonical_temp_storage_ctor_key(key) == owner
                )
            )
            if len(sites) > 1:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage.release cannot expose a region that merges "
                    "distinct constructor sites; construct the descriptor "
                    f"once before selecting its uses ({inst.loc})."
                )

    def _prepare_temp_storage_release_plans(self) -> None:
        rewrite = cast("CoopSinglePhaseRewrite", self)
        for inst, release in rewrite._temp_storage_releases.items():
            key = rewrite._canonical_temp_storage_ctor_key(release.ctor_key)
            rewrite._func_temp_storage_requirements.setdefault(
                key, _TempStorageRequirementSummary()
            )
            try:
                plan = rewrite._finalize_temp_storage_plan_for_var(key)
            except CoopSinglePhaseRewriteError as exc:
                raise CoopSinglePhaseRewriteError(
                    "TempStorage.release requires an existing nonempty region: "
                    f"{exc} ({inst.loc})."
                ) from exc
            if plan.size_in_bytes % release.itemsize:
                raise CoopSinglePhaseRewriteError(
                    f"TempStorage.release region size {plan.size_in_bytes} "
                    f"is not divisible by dtype item size {release.itemsize} "
                    f"({inst.loc})."
                )
            if plan.alignment % release.alignment:
                raise CoopSinglePhaseRewriteError(
                    f"TempStorage.release dtype requires {release.alignment}"
                    f"-byte alignment; region guarantees {plan.alignment} "
                    f"({inst.loc})."
                )

    def _emit_temp_storage_release(
        self, block: ir.Block, inst: ir.Assign
    ) -> None:
        rewrite = cast("CoopSinglePhaseRewrite", self)
        release = rewrite._temp_storage_releases[inst]
        plan = rewrite._finalize_temp_storage_plan_for_var(release.ctor_key)
        loc, scope = inst.loc, inst.target.scope
        byte_view = ir.Var(scope, _next_global_name("release_bytes"), loc)
        rewrite._emit_array_slice(
            block,
            source_var=cast(ir.Var, rewrite._temp_storage_backing_var),
            target_var=byte_view,
            start=plan.base_offset,
            stop=plan.base_offset + plan.size_in_bytes,
            loc=loc,
        )
        if release.dtype == uint8:
            block.append(ir.Assign(byte_view, inst.target, loc))
            return
        method = ir.Var(scope, _next_global_name("release_view"), loc)
        dtype_var = ir.Var(scope, _next_global_name("release_dtype"), loc)
        block.append(
            ir.Assign(ir.Expr.getattr(byte_view, "view", loc), method, loc)
        )
        block.append(
            ir.Assign(
                ir.Global(dtype_var.name, release.dtype, loc), dtype_var, loc
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.call(method, [dtype_var], (), loc), inst.target, loc
            )
        )
