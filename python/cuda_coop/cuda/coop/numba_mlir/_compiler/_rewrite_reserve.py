# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Plan typed shared-array reservations with their descriptor's scratch."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from numba_cuda_mlir.numba_cuda.core.errors import NumbaNotImplementedError
from numba_cuda_mlir.numba_cuda.np.numpy_support import as_dtype

from ._parameters import normalize_dtype_param
from ._rewrite_support import (
    _INFERENCE_EXCEPTIONS,
    CoopSinglePhaseRewriteError,
    _next_global_name,
    _TempStorageRequirementSummary,
    _TempStorageUseRequirement,
    ir,
)

if TYPE_CHECKING:
    from numba_cuda_mlir.numba_cuda.types import Type

    from ._rewrite import CoopSinglePhaseRewrite


@dataclass(frozen=True)
class _TempStorageReservation:
    ctor_key: str
    order: int
    num_elems: int
    dtype: Type
    size_in_bytes: int
    alignment: int


class _ReservationRewrite:
    def _collect_temp_storage_reservations(
        self, func_ir: ir.FunctionIR
    ) -> None:
        rewrite = cast("CoopSinglePhaseRewrite", self)
        rewrite._temp_storage_reservations = {}
        rewrite._temp_storage_reserve_methods = set()
        method_owners: dict[str, str] = {}
        blocks = [func_ir.blocks[label] for label in sorted(func_ir.blocks)]
        saved_block, saved_defs = rewrite._block, rewrite._block_defs
        try:
            for block in blocks:
                rewrite._block = block
                rewrite._block_defs = {
                    inst.target.name: inst.value
                    for inst in block.body
                    if isinstance(inst, ir.Assign)
                }
                for inst in block.body:
                    if not isinstance(inst, ir.Assign):
                        continue
                    value = inst.value
                    if not (
                        isinstance(value, ir.Expr)
                        and value.op == "getattr"
                        and value.attr == "reserve"
                    ):
                        continue
                    owner = rewrite._resolve_temp_storage_ctor_key(value.value)
                    if owner is None:
                        continue
                    definitions = func_ir._definitions.get(inst.target.name, ())
                    if len(definitions) != 1:
                        raise CoopSinglePhaseRewriteError(
                            "TempStorage.reserve method aliases "
                            "cannot be rebound."
                        )
                    method_owners[inst.target.name] = owner
                    rewrite._temp_storage_reserve_methods.add(inst)

            order = 0
            for block in blocks:
                rewrite._block = block
                rewrite._block_defs = {
                    inst.target.name: inst.value
                    for inst in block.body
                    if isinstance(inst, ir.Assign)
                }
                for inst in block.body:
                    order += 1
                    if not isinstance(inst, ir.Assign):
                        continue
                    call = inst.value
                    if not (
                        isinstance(call, ir.Expr)
                        and call.op == "call"
                        and call.func.name in method_owners
                    ):
                        continue
                    owner = rewrite._canonical_temp_storage_ctor_key(
                        method_owners[call.func.name]
                    )
                    specification = rewrite._temp_storage_ctor_specifications[
                        owner
                    ]
                    if specification.auto_sync:
                        raise CoopSinglePhaseRewriteError(
                            "TempStorage.reserve requires auto_sync=False; "
                            "callers must synchronize accesses to reserved "
                            "arrays explicitly."
                        )
                    rewrite._temp_storage_reservations[inst] = (
                        self._parse_temp_storage_reservation(call, owner, order)
                    )
        finally:
            rewrite._block, rewrite._block_defs = saved_block, saved_defs

        # A bound method may name several direct reserve calls. Passing it to
        # another function or forwarding it through IR aliases is unsupported.
        for block in blocks:
            for inst in block.body:
                used = {
                    var.name
                    for var in inst.list_vars()
                    if not (
                        isinstance(inst, ir.Assign)
                        and var.name == inst.target.name
                    )
                } & method_owners.keys()
                if not used:
                    continue
                if inst in rewrite._temp_storage_reservations and used == {
                    inst.value.func.name
                }:
                    continue
                raise CoopSinglePhaseRewriteError(
                    "TempStorage.reserve method aliases may only be called "
                    "directly; forwarding or escaping a reserve method "
                    "is unsupported."
                )

    def _parse_temp_storage_reservation(
        self, call: ir.Expr, owner: str, order: int
    ) -> _TempStorageReservation:
        rewrite = cast("CoopSinglePhaseRewrite", self)
        if (
            call.vararg is not None
            or getattr(call, "varkwarg", None) is not None
        ):
            raise CoopSinglePhaseRewriteError(
                "TempStorage.reserve does not accept *args or **kwargs."
            )
        if len(call.args) > 2:
            raise CoopSinglePhaseRewriteError(
                "TempStorage.reserve accepts num_elems and dtype positionally; "
                "alignment is keyword-only."
            )
        values = dict(zip(("num_elems", "dtype"), call.args))
        for name, value in call.kws:
            if name not in {"num_elems", "dtype", "alignment"}:
                raise CoopSinglePhaseRewriteError(
                    f"TempStorage.reserve got unexpected keyword {name!r}."
                )
            if name in values:
                raise CoopSinglePhaseRewriteError(
                    f"TempStorage.reserve got multiple values for {name!r}."
                )
            values[name] = value
        if "num_elems" not in values or "dtype" not in values:
            raise CoopSinglePhaseRewriteError(
                "TempStorage.reserve requires num_elems and dtype."
            )
        count_error = (
            "TempStorage.reserve num_elems must be a compile-time "
            "positive integer."
        )
        try:
            raw_count = rewrite._infer_constant(values["num_elems"])
            if isinstance(raw_count, bool):
                raise TypeError(count_error)
            num_elems = operator.index(raw_count)
            if num_elems <= 0:
                raise ValueError(count_error)
        except _INFERENCE_EXCEPTIONS as exc:
            raise CoopSinglePhaseRewriteError(count_error) from exc
        try:
            try:
                raw_dtype = rewrite._infer_constant(values["dtype"])
            except _INFERENCE_EXCEPTIONS:
                definition = rewrite._lookup_definition(values["dtype"])
                if not (
                    isinstance(definition, ir.Expr)
                    and definition.op == "getattr"
                    and definition.attr == "dtype"
                ):
                    raise ValueError("unresolved reservation dtype") from None
                raw_dtype = rewrite._resolve_var_dtype(definition.value)
            dtype = normalize_dtype_param(raw_dtype)
            numpy_dtype = as_dtype(dtype)
            # The compiler cannot reinterpret byte arrays as Boolean views.
            if numpy_dtype.kind not in "iufc" or numpy_dtype.itemsize <= 0:
                raise ValueError("unsupported reservation dtype")
        except (*_INFERENCE_EXCEPTIONS, NumbaNotImplementedError) as exc:
            raise CoopSinglePhaseRewriteError(
                "TempStorage.reserve dtype must be a compile-time fixed-size "
                "integer, floating-point, or complex dtype."
            ) from exc
        alignment = max(numpy_dtype.alignment, numpy_dtype.itemsize)
        if "alignment" in values:
            alignment_error = (
                "TempStorage.reserve alignment must be a compile-time positive "
                "power of 2, or None."
            )
            try:
                raw_alignment = rewrite._infer_constant(values["alignment"])
                if raw_alignment is not None:
                    if isinstance(raw_alignment, bool):
                        raise TypeError(alignment_error)
                    requested = operator.index(raw_alignment)
                    if requested <= 0 or requested & (requested - 1):
                        raise ValueError(alignment_error)
                    alignment = max(alignment, requested)
            except _INFERENCE_EXCEPTIONS as exc:
                raise CoopSinglePhaseRewriteError(alignment_error) from exc
        return _TempStorageReservation(
            ctor_key=owner,
            order=order,
            num_elems=num_elems,
            dtype=dtype,
            size_in_bytes=num_elems * numpy_dtype.itemsize,
            alignment=alignment,
        )

    def _add_temp_storage_reservations(
        self, requirements: dict[str, _TempStorageRequirementSummary]
    ) -> None:
        rewrite = cast("CoopSinglePhaseRewrite", self)
        for inst, reservation in rewrite._temp_storage_reservations.items():
            owner = rewrite._canonical_temp_storage_ctor_key(
                reservation.ctor_key
            )
            summary = requirements.setdefault(
                owner, _TempStorageRequirementSummary()
            )
            summary.max_size_in_bytes = max(
                summary.max_size_in_bytes, reservation.size_in_bytes
            )
            summary.max_alignment = max(
                summary.max_alignment, reservation.alignment
            )
            summary.uses.append(
                _TempStorageUseRequirement(
                    call_assign=inst,
                    order=reservation.order,
                    size_in_bytes=reservation.size_in_bytes,
                    alignment=reservation.alignment,
                    reservation=True,
                )
            )

    def _emit_temp_storage_reservation(
        self, block: ir.Block, inst: ir.Assign
    ) -> None:
        rewrite = cast("CoopSinglePhaseRewrite", self)
        reservation = rewrite._temp_storage_reservations[inst]
        plan = rewrite._finalize_temp_storage_plan_for_var(reservation.ctor_key)
        region = plan.slices_by_call_id[id(inst)]
        backing = rewrite._temp_storage_backing_var
        assert backing is not None
        loc, scope = inst.loc, inst.target.scope
        byte_view = ir.Var(scope, _next_global_name("reserve_bytes"), loc)
        rewrite._emit_array_slice(
            block,
            source_var=backing,
            target_var=byte_view,
            start=plan.base_offset + region.offset,
            stop=plan.base_offset + region.offset + region.size_in_bytes,
            loc=loc,
        )
        view_method = ir.Var(scope, _next_global_name("reserve_view"), loc)
        dtype = ir.Var(scope, _next_global_name("reserve_dtype"), loc)
        block.append(
            ir.Assign(ir.Expr.getattr(byte_view, "view", loc), view_method, loc)
        )
        block.append(
            ir.Assign(
                ir.Global(
                    _next_global_name("reserve_type"), reservation.dtype, loc
                ),
                dtype,
                loc,
            )
        )
        block.append(
            ir.Assign(
                ir.Expr.call(view_method, [dtype], (), loc), inst.target, loc
            )
        )
