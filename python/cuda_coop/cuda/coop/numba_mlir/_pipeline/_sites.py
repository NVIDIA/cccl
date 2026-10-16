# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Find a kernel's cooperative call sites and check them, from typed IR.

Site analysis is the first stage of ``CoopCallResolution``. It reads the
typed function and nothing else: not the launch, not NVRTC. It does not
change the IR. For each kernel it:

* collects the cooperative call sites in program order;
* refuses cooperative calls inside functions compiled as separate device
  functions (``targetoptions["device"]``), which have no launch;
* reads the extent of a ``cuda.coop.numba_mlir`` call's ``cuda.local.array``
  payload from its constructor's shape, as coop's planner does;
* binds each site's ``valid_items``, ``oob_default`` and ``offset`` as
  static or runtime by where the value comes from, as coop's planner does,
  and checks them and any scalar ``store`` value by coop's rules;
* builds each site's dims-free request with its controls;
* measures the kernel's own static shared memory.

These checks run before anything is specialized or compiled. Native typing
has already resolved payload dtypes and checked indexed writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from numbers import Integral
from typing import Any

from . import _bridges as nb
from ._errors import TypedPipelineError
from ._templates import (
    CONTROLS,
    OPERATIONS,
    QUALIFIED_OPERATIONS,
    RESULT_EXTENT_SOURCES,
    OperationRequest,
    is_scalar_value,
    is_wrapper_location,
    request_for,
    with_controls,
)
from ._types import ThreadDataType


@dataclass(frozen=True)
class Scalar:
    """A scalar ``store`` value; ``value`` is ``None`` when it is runtime."""

    value: Any


@dataclass(frozen=True, eq=False)
class Site:
    """A call site; ``line`` is the kernel source line of the public call.

    ``stmt`` is the call's assignment; its target names the site.
    """

    stmt: Any
    kind: str
    line: int | None
    request: OperationRequest
    scalar: Scalar | None = None


@dataclass(frozen=True)
class Sites:
    """A kernel's checked call sites, in program order.

    ``user_shared_bytes`` is the static shared memory the kernel allocates
    itself with ``cuda.shared.array``.
    """

    sites: tuple[Site, ...]
    user_shared_bytes: int
    user_dynamic_shared_at: str | None = None


_RUNTIME = object()


# Checks -----------------------------------------------------------------------


def _callee(typemap: dict, expr: Any) -> Any:
    ty = typemap.get(expr.func.name)
    if isinstance(ty, nb.types.Function):
        return ty.typing_key
    return None


def _shape(expr: Any) -> Any:
    return expr.args[0] if expr.args else dict(expr.kws).get("shape")


def _shared_constant(func_ir: Any, typemap: dict, var: Any) -> Any:
    if var is None:
        return None
    ty = typemap[var.name]
    if isinstance(ty, nb.types.IntegerLiteral):
        return ty.literal_value
    try:
        return func_ir.infer_constant(var.name)
    except nb.errors.ConstantInferenceError:
        return None


def _user_shared_memory(
    func_ir: Any, typemap: dict, calltypes
) -> tuple[int, str | None]:
    """Count static allocations and retain a conflicting dynamic location.

    Shared shapes are compiler constants, including tuples. A zero or
    nonconstant shape uses the compiler's dynamic shared-memory window and
    must remain distinguishable from a kernel with no shared allocations.
    """

    total = 0
    dynamic_at = None
    for label in sorted(func_ir.blocks):
        for stmt in func_ir.blocks[label].body:
            if not (
                isinstance(stmt, nb.ir.Assign)
                and isinstance(stmt.value, nb.ir.Expr)
                and stmt.value.op == "call"
                and _callee(typemap, stmt.value) is nb.shared_array
            ):
                continue
            call = stmt.value
            shape = _shared_constant(func_ir, typemap, _shape(call))
            dimensions = shape if isinstance(shape, tuple) else (shape,)
            if not dimensions or not all(
                isinstance(extent, Integral) and int(extent) > 0
                for extent in dimensions
            ):
                dynamic_at = dynamic_at or str(stmt.loc)
                continue
            kwargs = dict(call.kws)
            alignment_var = (
                call.args[2] if len(call.args) > 2 else kwargs.get("alignment")
            )
            alignment = _shared_constant(func_ir, typemap, alignment_var)
            if alignment is None:
                alignment = _shared_constant(
                    func_ir, typemap, kwargs.get("alignas")
                )
            # Numba-CUDA-MLIR gives ordinary shared arrays an 8-byte alignment.
            alignment = 8 if alignment is None else alignment
            if isinstance(alignment, Integral) and int(alignment) > 0:
                alignment = int(alignment)
                total = -(-total // alignment) * alignment
            result = calltypes[call].return_type
            # Numba types bool without a bitwidth and stores it in a byte.
            bitwidth = getattr(result.dtype, "bitwidth", 8)
            total += prod(int(extent) for extent in dimensions) * bitwidth // 8
    return total, dynamic_at


# The analysis -----------------------------------------------------------------


def _resolved_request(request: OperationRequest | None) -> OperationRequest:
    assert request is not None, (
        "native typing left a cooperative request unresolved"
    )
    return request


class _Analysis:
    """The typed function, and the per-site reads that need its IR."""

    def __init__(self, func_ir, typemap, args):
        self.func_ir, self.typemap, self.args = func_ir, typemap, tuple(args)

    def collect(self) -> list[tuple[Any, str, int | None, bool]]:
        found = []
        line = None
        for label in sorted(self.func_ir.blocks):
            for stmt in self.func_ir.blocks[label].body:
                if not is_wrapper_location(stmt.loc):
                    line = stmt.loc.line
                if (
                    isinstance(stmt, nb.ir.Assign)
                    and isinstance(stmt.value, nb.ir.Expr)
                    and stmt.value.op == "call"
                ):
                    callee = _callee(self.typemap, stmt.value)
                    kind = OPERATIONS.get(callee)
                    if kind is not None:
                        found.append(
                            (stmt, kind, line, callee in QUALIFIED_OPERATIONS)
                        )
        return found

    def site(self, stmt, kind, line, qualified) -> Site:
        expr = stmt.value
        if kind in {"run_length_decode", "run_length_decode_into"}:
            from . import _run_length

            return Site(
                stmt,
                kind,
                line,
                _resolved_request(
                    _run_length.site_request(self, stmt, kind, qualified)
                ),
            )
        if kind == "histogram":
            from . import _histogram

            return Site(
                stmt,
                kind,
                line,
                _resolved_request(
                    _histogram.site_request(self, stmt, kind, qualified)
                ),
            )
        if kind in {"adjacent_difference", "discontinuity"}:
            from . import _neighbors

            return Site(
                stmt,
                kind,
                line,
                _resolved_request(
                    _neighbors.site_request(self, stmt, kind, qualified)
                ),
            )
        if kind in {
            "topk_min_keys",
            "topk_min_pairs",
            "topk_max_keys",
            "topk_max_pairs",
        }:
            from . import _topk

            return Site(
                stmt,
                kind,
                line,
                _resolved_request(
                    _topk.site_request(self, stmt, kind, qualified)
                ),
            )
        if kind in {"radix_sort_keys", "radix_sort_pairs", "radix_rank_keys"}:
            from . import _radix

            return Site(
                stmt,
                kind,
                line,
                _resolved_request(
                    _radix.site_request(self, stmt, kind, qualified)
                ),
            )
        if kind in {"merge_sort_keys", "merge_sort_pairs"}:
            from . import _merge_sort

            return Site(
                stmt,
                kind,
                line,
                _resolved_request(
                    _merge_sort.site_request(self, stmt, kind, qualified)
                ),
            )
        if kind in {
            "scan",
            "inclusive_scan",
            "exclusive_scan",
            "inclusive_sum",
            "exclusive_sum",
        }:
            from . import _scan

            value_type = self.typemap[expr.args[1].name]
            dtype = getattr(value_type, "dtype", nb.unliteral(value_type))
            operands = {"value": expr.args[1], **dict(expr.kws)}
            if len(expr.args) > 2:
                operands["prefix_state"] = expr.args[2]
            request = _scan.request_for(
                kind,
                [self.typemap[a.name] for a in expr.args],
                {k: self.typemap[v.name] for k, v in expr.kws},
                qualified=qualified,
                extents={
                    name: self._value_extent(operands[name], kind)
                    for name in ("value", "prefix_state", "aggregate_output")
                    if name in operands
                },
                controls=_scan.bind_controls(self, stmt, dtype),
            )
            if request is None:
                raise TypedPipelineError(
                    f"coop.{kind}: payload dtype is unresolved"
                )
            return Site(stmt, kind, line, request)
        if kind in {"exchange", "shuffle"}:
            from . import _exchange

            value_type = self.typemap[expr.args[1].name]
            dtype = getattr(value_type, "dtype", nb.unliteral(value_type))
            operands = {"value": expr.args[1], **dict(expr.kws)}
            request = _exchange.request_for(
                kind,
                [self.typemap[a.name] for a in expr.args],
                {k: self.typemap[v.name] for k, v in expr.kws},
                qualified=qualified,
                extents={
                    name: self._value_extent(operands[name], kind)
                    for name in ("value", "ranks", "valid_flags")
                    if name in operands
                },
                controls=self._controls(stmt, kind, dtype),
            )
            if request is None:
                raise TypedPipelineError(
                    f"coop.{kind}: payload dtype is unresolved"
                )
            return Site(stmt, kind, line, request)
        if kind in {"reduce", "sum"}:
            from . import _reductions

            value_type = self.typemap[expr.args[1].name]
            dtype = getattr(value_type, "dtype", nb.unliteral(value_type))
            request = _reductions.request_for(
                kind,
                [self.typemap[a.name] for a in expr.args],
                {k: self.typemap[v.name] for k, v in expr.kws},
                qualified=qualified,
                extent=self._array_extent(stmt, kind, index=1),
                controls=self._controls(stmt, kind, dtype),
            )
            if request is None:
                raise TypedPipelineError(
                    f"coop.{kind}: payload extent is not static"
                )
            return Site(stmt, kind, line, request)
        request = request_for(
            kind,
            [self.typemap[a.name] for a in expr.args],
            {k: self.typemap[v.name] for k, v in expr.kws},
            qualified=qualified,
            extent=self._array_extent(stmt, kind) if qualified else None,
        )
        scalar = self._store_scalar(stmt, kind, request.dtype)
        request = with_controls(
            request, self._controls(stmt, kind, request.dtype)
        )
        return Site(stmt, kind, line, request, scalar)

    def _provenance(self, var: Any) -> Any:
        """Return ``var``'s static provenance, or ``_RUNTIME``.

        A value is static when coop's planner would call it static: it
        comes, through aliases, casts and phis, from a constant, a global, a
        free variable or a literal argument.
        """

        from .._compiler._scalar_provenance import (
            try_resolve_static_scalar_provenance,
        )

        definitions = self.func_ir._definitions
        arguments = self.args
        resolved, scalar = try_resolve_static_scalar_provenance(
            var,
            definitions=lambda v: definitions.get(v.name, ()),
            argument_type=lambda index: (
                arguments[index] if 0 <= index < len(arguments) else None
            ),
        )
        return scalar if resolved else _RUNTIME

    def _controls(self, stmt, kind: str, dtype: Any) -> tuple:
        """Bind the site's controls as coop's planner binds them.

        Checks follow the planner's ``_rewrite_load_store``: static
        ``valid_items`` and ``offset`` first, then the integer types of
        runtime ones, then ``oob_default`` against the payload dtype.
        """

        from ..._core import ArgumentBinding
        from .._compiler._parameters import (
            _validate_common_numeric_dtype,
            _validate_runtime_integer_dtype,
            coerce_static_scalar,
        )

        keywords = dict(stmt.value.kws)
        found = {}
        for name in CONTROLS[kind]:
            var = keywords.get(name)
            if var is None:
                continue
            scalar = self._provenance(var)
            if scalar is _RUNTIME:
                found[name] = (_RUNTIME, self.typemap[var.name])
            elif scalar.value is not None:
                found[name] = (scalar, None)

        def check(condition: bool, message: str) -> None:
            if condition:
                raise TypedPipelineError(message)

        for name in ("valid_items", "offset"):
            scalar, _ = found.get(name, (None, None))
            if scalar is None or scalar is _RUNTIME:
                continue
            value = scalar.value
            check(
                isinstance(value, bool) or not isinstance(value, Integral),
                f"coop {kind} {name} must be an integer",
            )
            if name == "offset":
                check(
                    int(value) < 0,
                    "coop load/store static offset must be nonnegative",
                )
                check(
                    int(value) > (1 << 63) - 1,
                    "coop load/store static offset must fit a signed 64-bit "
                    "integer",
                )
        for name in ("valid_items", "offset"):
            scalar, ty = found.get(name, (None, None))
            if scalar is _RUNTIME:
                try:
                    _validate_runtime_integer_dtype(
                        ty, operation=kind, parameter=name
                    )
                except TypeError as exc:
                    raise TypedPipelineError(str(exc)) from None

        controls = []
        for name in CONTROLS[kind]:
            if name not in found:
                continue
            scalar, ty = found[name]
            if scalar is _RUNTIME:
                if name == "oob_default":
                    try:
                        ty = _validate_common_numeric_dtype(
                            ty, operation="load", parameter="oob_default"
                        )
                    except (TypeError, ValueError) as exc:
                        raise TypedPipelineError(str(exc)) from None
                    check(
                        ty != dtype,
                        "cuda.coop.numba_mlir.load runtime oob_default dtype "
                        f"{ty} does not match payload dtype {dtype}",
                    )
                controls.append((name, ArgumentBinding.runtime()))
                continue
            value = scalar.value
            if name == "oob_default":
                try:
                    value = coerce_static_scalar(
                        value,
                        dtype,
                        operation="load",
                        parameter="oob_default",
                        source_dtype=scalar.dtype,
                    )
                except (TypeError, ValueError) as exc:
                    raise TypedPipelineError(str(exc)) from None
            controls.append((name, ArgumentBinding.static(value)))
        return tuple(controls)

    def _store_scalar(self, stmt, kind: str, dtype: Any) -> Scalar | None:
        """Check a scalar ``store`` value as coop's planner does.

        A static value converts to the destination's dtype under coop's
        scalar rules; a runtime value must already have that dtype exactly.
        """

        from .._compiler._parameters import coerce_static_scalar

        if kind != "store":
            return None
        var = stmt.value.args[2]
        ty = self.typemap[var.name]
        if not is_scalar_value(ty):
            return None
        scalar = self._provenance(var)
        if scalar is _RUNTIME or scalar.value is None:
            actual = getattr(ty, "literal_type", ty)
            if actual != dtype:
                raise TypedPipelineError(
                    f"cuda.coop.numba_mlir.store value dtype {actual} does "
                    f"not match destination dtype {dtype}"
                )
            return Scalar(None)
        try:
            value = coerce_static_scalar(
                scalar.value,
                dtype,
                operation="store",
                parameter="value",
                source_dtype=scalar.dtype,
            )
        except (TypeError, ValueError) as exc:
            raise TypedPipelineError(str(exc)) from None
        return Scalar(value)

    def _array_extent(self, stmt, kind: str, index: int = 2) -> int | None:
        return self._value_extent(stmt.value.args[index], kind)

    def _value_extent(self, var, kind: str) -> int | None:
        """Recover static shape through array results and tuple projections.

        Registry entries are a source argument index, a tuple describing
        result slots, or a resolver of ``(analysis, call)`` to an extent.
        Element dtypes remain owned by native typing.
        """

        ty = self.typemap[var.name]
        if not isinstance(ty, nb.types.Array) or isinstance(ty, ThreadDataType):
            return None
        extents, seen = set(), set()
        pending: list[tuple[Any, tuple[int, ...]]] = [(var, ())]
        while pending:
            var, projection = pending.pop()
            key = (var.name, projection)
            if key in seen:
                continue
            seen.add(key)
            for definition in self.func_ir._definitions.get(var.name, ()):
                if isinstance(definition, nb.ir.Var):
                    pending.append((definition, projection))
                    continue
                if isinstance(definition, nb.ir.Expr):
                    if definition.op in {"cast", "exhaust_iter"}:
                        pending.append((definition.value, projection))
                        continue
                    if definition.op == "phi":
                        pending.extend(
                            (value, projection)
                            for value in definition.incoming_values
                        )
                        continue
                    if definition.op == "static_getitem":
                        pending.append(
                            (definition.value, (definition.index, *projection))
                        )
                        continue
                    if definition.op == "build_tuple" and projection:
                        pending.append(
                            (definition.items[projection[0]], projection[1:])
                        )
                        continue
                    if definition.op == "call":
                        source = RESULT_EXTENT_SOURCES.get(
                            _callee(self.typemap, definition)
                        )
                        if callable(source):
                            extent = source(self, definition)
                            rest = projection
                            while isinstance(extent, tuple) and rest:
                                extent, rest = extent[rest[0]], rest[1:]
                            extents.add(
                                extent
                                if isinstance(extent, int) and not rest
                                else None
                            )
                            continue
                        rest = projection
                        while isinstance(source, tuple) and rest:
                            source, rest = source[rest[0]], rest[1:]
                        if isinstance(source, int) and not rest:
                            pending.append((definition.args[source], ()))
                            continue
                extents.add(
                    None if projection else self._local_array_extent(definition)
                )
        if None in extents or not extents:
            raise TypedPipelineError(
                f"cuda.coop.numba_mlir.{kind} requires a static "
                "items_per_thread extent before provider selection"
            )
        if len(extents) > 1:
            raise TypedPipelineError(
                "cuda.coop.numba_mlir loop-carried payloads have "
                "inconsistent items_per_thread extents"
            )
        return extents.pop()

    def _local_array_extent(self, definition: Any) -> int | None:
        if not (
            isinstance(definition, nb.ir.Expr)
            and definition.op == "call"
            and _callee(self.typemap, definition) is nb.local_array
        ):
            return None
        shape = _shape(definition)
        ty = None if shape is None else self.typemap.get(shape.name)
        if isinstance(ty, nb.types.IntegerLiteral):
            return ty.literal_value
        return None


def analyze(
    func_ir: Any,
    typemap: dict,
    calltypes: Any,
    *,
    args: tuple,
    targetoptions: dict,
    func_id: Any,
) -> Sites | None:
    """Return the kernel's checked sites, or ``None`` for a device function.

    A device function without cooperative calls has nothing to resolve; one
    with them is an error.
    """

    analysis = _Analysis(func_ir, typemap, args)
    found = analysis.collect()
    if targetoptions.get("device"):
        if found:
            _, kind, line, _ = found[0]
            raise TypedPipelineError(
                f"coop.{kind} at line {line} "
                f"is inside {func_id.func_qualname!r}, a separately "
                "compiled device function, which has no launch to plan "
                "for. Let the function inline (the cuda.jit default) "
                "instead of inline='never'."
            )
        return None
    sites = tuple(analysis.site(*f) for f in found)
    user_shared_bytes, dynamic_at = _user_shared_memory(
        func_ir, typemap, calltypes
    )
    return Sites(sites, user_shared_bytes, dynamic_at)


__all__ = ["Scalar", "Site", "Sites", "analyze"]
