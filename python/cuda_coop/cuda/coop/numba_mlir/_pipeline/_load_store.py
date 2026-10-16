# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete Load/Store providers and assign each Warp its memory tile."""

from __future__ import annotations

import operator
from typing import cast

from . import _bridges as nb
from ._errors import TypedPipelineError

EFFECTIVE_OFFSET_KEYWORD = "__coop_effective_offset"


def complete(request, block, plan):
    from ..._core import ArgumentBinding, GroupLoweringTarget, StorageOwnership
    from .._lowering import _load_store
    from .._lowering._core import NumbaMlirCoreAdapter

    storage_free = plan.temp_storage.ownership is StorageOwnership.NONE
    kwargs = {"threads_per_block": block}
    if plan.target is GroupLoweringTarget.CUB_BLOCK:
        name = request.kind
    elif plan.target is GroupLoweringTarget.CUB_WARP:
        topology = plan.topology
        if (
            topology is None
            or topology.group_kind not in {"warp", "threads_within_warp"}
            or topology.logical_width not in {1, 2, 4, 8, 16, 32}
        ):
            raise TypedPipelineError(
                "Warp Load/Store requires a power-of-two width in [1, 32]"
            )
        name = f"warp_{request.kind}"
        kwargs["threads_in_warp"] = topology.logical_width
    else:
        raise TypedPipelineError(
            f"coop.{request.kind}: unsupported target {plan.target.value}"
        )
    if not storage_free:
        name = f"_{name}_with_storage"
    implementation = plan.implementation
    omitted = ArgumentBinding.omitted()
    kwargs.update(
        dtype=NumbaMlirCoreAdapter().normalize_dtype(
            implementation.template_arguments["T"]
        ),
        items_per_thread=request.items_per_thread,
        algorithm=implementation.metadata["algorithm"],
        num_valid_items=request.binding("valid_items") or omitted,
        offset=(
            ArgumentBinding.runtime()
            if requires_tile_offset(plan)
            else request.binding("offset") or omitted
        ),
    )
    if request.binding("oob_default") is not None:
        kwargs["oob_default"] = request.binding("oob_default")
    return getattr(_load_store, name), kwargs


def requires_tile_offset(plan):
    return bool(
        plan.implementation.metadata.get(
            "requires_runtime_effective_offset", False
        )
    )


def materialize_offsets(
    func_ir, typemap, calltypes, typingctx, sites, specialization
):
    """Pass the provider each group's tile origin plus the user's offset.

    The core plan has checked topology, group participation and static
    offset range. Runtime offset bounds remain its caller precondition.
    This also runs for storage-free algorithms, which have no scratch IR.
    """

    from ..._core import BindingKind
    from ._materialize import _StorageIR

    if not specialization.lowering_plans:
        return
    planned = {
        site.stmt.target.name: (site, plan)
        for site, plan in zip(sites, specialization.lowering_plans, strict=True)
        if site.kind in {"load", "store"} and requires_tile_offset(plan)
    }
    if not planned:
        return
    for block in func_ir.blocks.values():
        rewritten = []
        for stmt in block.body:
            name = stmt.target.name if isinstance(stmt, nb.ir.Assign) else None
            entry = planned.get(name)
            if entry is not None:
                site, plan = entry
                emitter = _StorageIR(
                    typemap, calltypes, typingctx, block.scope, stmt.loc
                )
                origin = emitter.binary(
                    operator.mul,
                    emitter.instance_index(plan.topology),
                    emitter.constant(
                        plan.topology.logical_width
                        * site.request.items_per_thread
                    ),
                )
                binding = site.request.binding("offset")
                if binding is None or binding.kind is BindingKind.OMITTED:
                    base = emitter.constant(0)
                elif binding.kind is BindingKind.STATIC:
                    base = emitter.constant(binding.value)
                else:
                    base = dict(cast(nb.ir.Expr, stmt.value).kws)["offset"]
                effective = emitter.binary(operator.add, origin, base)
                call = cast(nb.ir.Expr, stmt.value)
                call.kws = (
                    *call.kws,
                    (EFFECTIVE_OFFSET_KEYWORD, effective),
                )
                rewritten.extend(emitter.body)
            rewritten.append(stmt)
        block.body = rewritten
    func_ir._definitions = nb.build_definitions(func_ir.blocks)
