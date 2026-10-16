# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type cooperative calls with Numba typing templates.

Each kernel-callable object of both public APIs, ``cuda.coop`` and
``cuda.coop.numba_mlir``, gets one typing template. Objects the typed
pipeline supports are typed for real; every other one gets a template that
rejects the call with ``unsupported_primitive``, so a kernel never falls
back to the planner and never fails with Numba's generic "untyped global"
message.

``load`` and ``store`` are reached through signature-preserving inline
wrappers. Native cooperative constraints refine payloads from registered
producer and Store contracts. The private operation types with a precise
payload and builds a dims-free ``GroupPrimitiveCall`` that call resolution
completes with the launch's block.

Each API has its own private operations, because their payload rules
differ: a ``cuda.coop.numba_mlir`` call also accepts a ``cuda.local.array``
payload, whose extent call resolution reads from the IR, and a
``cuda.coop`` call refuses one with coop's common-API diagnostic.

A ``ThreadData``'s ``items_per_thread`` attribute is its extent as an
integer literal, the constant coop's planner substitutes for it.
"""

from __future__ import annotations

import inspect
import linecache
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from . import _bridges as nb
from ._errors import TypedPipelineError, unsupported_primitive
from ._types import (
    EXPLICIT,
    CoopGroupType,
    ThreadDataType,
)

# Public names that are never called inside a kernel: host-side descriptors
# and registration, typing protocols, and Numba-CUDA-MLIR's own namespaces
# re-exported for convenience.
_NOT_KERNEL_CALLABLE = frozenset(
    {
        "Hierarchy",
        "TempStorageLike",
        "ThreadDataLike",
        "ThreadGroup",
        "ThreadHierarchy",
        "local",
        "register",
        "shared",
    }
)

# Optional scalar controls, in the order coop's providers take them at run
# time.
CONTROLS = {
    "load": ("valid_items", "oob_default", "offset"),
    "store": ("valid_items", "offset"),
    "exchange": (),
    "shuffle": ("distance",),
    "reduce": ("valid_items",),
    "sum": ("valid_items",),
}

# Static extent provenance for qualified array-valued operation results.
RESULT_EXTENT_SOURCES: dict[object, int] = {}

_PAYLOAD = {"load": "output", "store": "value"}
_MEMORY = {"load": "source", "store": "destination"}


def kernel_primitives() -> dict[Callable, str]:
    """Map each kernel-callable public object to its qualified name."""

    from ... import numba_mlir as qualified
    from ..._core import api as common

    primitives: dict[Callable, str] = {}
    for prefix, module in (
        ("cuda.coop", common),
        ("cuda.coop.numba_mlir", qualified),
    ):
        for name in module.__all__:
            if name in _NOT_KERNEL_CALLABLE:
                continue
            obj = getattr(module, name)
            if callable(obj):
                primitives.setdefault(obj, f"{prefix}.{name}")
    return primitives


def group_operation(obj: object) -> str | None:
    """Return coop's registered group operation for a public object."""

    from ..._core.api._dispatch import _common_group_operation_name
    from .._compiler._operations import group_operation_name

    return _common_group_operation_name(obj) or group_operation_name(obj)


def register_template(
    key: object,
    generic: Callable,
    name: str,
    *,
    prefer_literal: bool = False,
) -> type:
    """Type calls to ``key`` with ``generic(template, args, kws)``."""

    template = type(
        f"CoopTyped_{name}",
        (nb.AbstractTemplate,),
        {"key": key, "generic": generic, "prefer_literal": prefer_literal},
    )
    nb.typing_registry.register(template)
    nb.typing_registry.register_global(key, nb.types.Function(template))
    return template


def _rejecting(qualified_name: str) -> Callable:
    def generic(self, args, kws):
        raise unsupported_primitive(qualified_name)

    return generic


def _not_inlined(qualified_name: str) -> Callable:
    def generic(self, args, kws):
        raise TypedPipelineError(
            f"{qualified_name} reached typing without its inline wrapper. "
            "Call it directly by name inside the kernel."
        )

    return generic


def _omitted(ty: Any) -> bool:
    return ty is None or isinstance(ty, (nb.types.NoneType, nb.types.Omitted))


def _payload_dtype(ty: Any) -> Any:
    if isinstance(ty, nb.types.NumberClass):
        return ty.instance_type
    if isinstance(ty, nb.types.DType):
        return ty.dtype
    return None


# ThreadData ----------------------------------------------------------------

_CONSTRUCTOR_PARAMETERS = ("items_per_thread", "dtype", "alignment")


def _bind_constructor(qualified_name: str, args, kws) -> dict:
    if len(args) > 2:
        raise TypedPipelineError(
            f"{qualified_name}: alignment is a keyword-only argument"
        )
    bound = dict(zip(_CONSTRUCTOR_PARAMETERS, args))
    for name, ty in kws.items():
        if name not in _CONSTRUCTOR_PARAMETERS or name in bound:
            raise TypedPipelineError(
                f"{qualified_name}: unexpected argument {name!r}"
            )
        bound[name] = ty
    if "items_per_thread" not in bound:
        raise TypedPipelineError(
            f"{qualified_name}: items_per_thread is required"
        )
    return bound


def _fold_constructor_arguments(args, kws) -> tuple:
    bound = dict(zip(_CONSTRUCTOR_PARAMETERS, args))
    bound.update(dict(kws))
    return tuple(bound[n] for n in _CONSTRUCTOR_PARAMETERS if n in bound)


def constructor_alignment(qualified_name: str, ty: Any) -> int | None:
    """Return the normalized ``alignment=`` of a constructor call."""

    from ..._core.api._payload import _normalize_alignment

    if _omitted(ty):
        return None
    if not isinstance(ty, nb.types.IntegerLiteral):
        raise TypedPipelineError(
            f"{qualified_name}: alignment must be a compile-time integer"
        )
    try:
        return _normalize_alignment(ty.literal_value)
    except (TypeError, ValueError) as exc:
        raise TypedPipelineError(f"{qualified_name}: {exc}") from None


def _thread_data(qualified_name: str) -> Callable:
    def generic(self, args, kws):
        bound = _bind_constructor(qualified_name, args, kws)
        items = bound["items_per_thread"]
        if isinstance(items, nb.types.Integer) and not isinstance(
            items, nb.types.IntegerLiteral
        ):
            raise nb.force_literal_arg({0}, _fold_constructor_arguments)
        if (
            not isinstance(items, nb.types.IntegerLiteral)
            or items.literal_value <= 0
        ):
            raise TypedPipelineError(
                f"{qualified_name}: items_per_thread must be a positive "
                f"compile-time integer, got {items}"
            )
        alignment = constructor_alignment(
            qualified_name, bound.get("alignment")
        )
        dtype_ty = bound.get("dtype")
        if _omitted(dtype_ty):
            payload = ThreadDataType(
                nb.undefined, items.literal_value, alignment=alignment
            )
        else:
            dtype = _payload_dtype(dtype_ty)
            if dtype is None:
                raise TypedPipelineError(
                    f"{qualified_name}: dtype must be a Numba or NumPy "
                    f"scalar type, got {dtype_ty}"
                )
            payload = ThreadDataType(
                dtype, items.literal_value, EXPLICIT, alignment
            )
        return nb.signature(payload, *args, *kws.values())

    return generic


class _ThreadDataAttributes(nb.AttributeTemplate):
    key = ThreadDataType

    def resolve_items_per_thread(self, td):
        return nb.types.IntegerLiteral(td.items)


# load and store --------------------------------------------------------------


def _coop_load_op(*args, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _coop_store_op(*args, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _coop_qualified_load_op(*args, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


def _coop_qualified_store_op(*args, **kwargs):
    raise NotImplementedError("cuda.coop: kernel only")


OPERATIONS: dict[Callable, str] = {
    _coop_load_op: "load",
    _coop_store_op: "store",
    _coop_qualified_load_op: "load",
    _coop_qualified_store_op: "store",
}

# The private operations of ``cuda.coop.numba_mlir``'s load and store.
QUALIFIED_OPERATIONS = {_coop_qualified_load_op, _coop_qualified_store_op}

_QUALIFIED_PREFIX = "cuda.coop.numba_mlir."


class OperationRequest(Protocol):
    """Resolved family contract consumed by whole-kernel stages."""

    @property
    def kind(self) -> str: ...

    @property
    def dtype(self) -> Any: ...

    @property
    def items_per_thread(self) -> int: ...

    @property
    def algorithm(self) -> str | None: ...

    @property
    def group(self) -> Any: ...

    @property
    def controls(self) -> tuple: ...

    @property
    def call(self) -> Any: ...

    @property
    def key(self) -> tuple: ...


@dataclass(frozen=True, eq=False)
class LoadStoreRequest:
    """A typed ``load`` or ``store``, before the launch is known.

    ``controls`` holds coop's ``ArgumentBinding`` for each control the call
    supplies, in ``CONTROLS`` order. A static binding carries its value and
    a runtime one does not, so two calls that differ only in a runtime
    value have the same request, and two that differ in a static value do
    not. Requests are interned by ``key``, so identity is equality.

    ``call`` is coop's ``GroupPrimitiveCall``; call resolution completes
    it with ``plan_group_primitive`` once it knows the block.
    """

    kind: str
    dtype: Any
    items_per_thread: int
    algorithm: str
    group: Any
    controls: tuple[tuple[str, Any], ...]
    call: Any
    storage: tuple = ()

    @property
    def key(self) -> tuple:
        return _request_key(
            self.group,
            self.kind,
            self.dtype,
            self.items_per_thread,
            self.algorithm,
            self.controls,
            self.storage,
        )

    def binding(self, name: str) -> Any:
        """Return the binding of control ``name``, or ``None`` if omitted."""

        return dict(self.controls).get(name)


_REQUESTS: dict[tuple, LoadStoreRequest] = {}


def _request_key(
    group, kind, dtype, items, algorithm, controls, storage
) -> tuple:
    return (
        group.key,
        kind,
        dtype,
        items,
        algorithm,
        tuple((name, b.semantic_key) for name, b in controls),
        storage,
    )


def storage_options(ty: Any) -> dict:
    """Return core storage semantics without allocation-local identities."""

    if _omitted(ty):
        return {}
    from ._temp_storage import TempStorageType

    if not isinstance(ty, TempStorageType):
        raise TypedPipelineError(
            "temp_storage must be a TempStorage descriptor"
        )
    return ty.semantics_kwargs


def _intern(kind, group, dtype, items, algorithm, controls=(), storage=()):
    from ..._core import ArgumentBinding
    from ..._core.group._model import GroupPrimitiveCall
    from ..._core.group.load_store import (
        GroupLoadStoreAlgorithm,
        GroupLoadStoreKind,
        GroupLoadStoreSemantics,
    )
    from .._lowering._core import NumbaMlirCoreAdapter

    key = _request_key(group, kind, dtype, items, algorithm, controls, storage)
    request = _REQUESTS.get(key)
    if request is not None:
        return request
    bindings = dict(controls)
    omitted = ArgumentBinding.omitted()
    try:
        semantics = GroupLoadStoreSemantics(
            kind=GroupLoadStoreKind(kind),
            dtype=NumbaMlirCoreAdapter().core_dtype(dtype),
            items_per_thread=items,
            algorithm=GroupLoadStoreAlgorithm(algorithm),
            valid_items=bindings.get("valid_items", omitted),
            oob_default=bindings.get("oob_default", omitted),
            offset=bindings.get("offset", omitted),
            **dict(storage),
        )
    except (TypeError, ValueError) as exc:
        raise TypedPipelineError(f"coop.{kind}: {exc}") from None
    call = GroupPrimitiveCall(group=group.group, operation=semantics)
    request = _REQUESTS[key] = LoadStoreRequest(
        kind, dtype, items, algorithm, group, tuple(controls), call, storage
    )
    return request


def with_controls(
    request: LoadStoreRequest, controls: tuple[tuple[str, Any], ...]
) -> LoadStoreRequest:
    """Return the request for ``request``'s transfer with ``controls``.

    Raises the coop diagnostic for a combination coop rejects, such as
    ``oob_default`` without ``valid_items``.
    """

    return _intern(
        request.kind,
        request.group,
        request.dtype,
        request.items_per_thread,
        request.algorithm,
        controls,
        request.storage,
    )


def _algorithm(kind: str, ty: Any) -> str | None:
    from ..._core.group.load_store import GroupLoadStoreAlgorithm

    if _omitted(ty):
        return "direct"
    if not isinstance(ty, nb.types.StringLiteral):
        if isinstance(ty, nb.types.UnicodeType):
            return None
        raise TypedPipelineError(
            f"coop.{kind}: algorithm must be a string literal, got {ty}"
        )
    try:
        return GroupLoadStoreAlgorithm(ty.literal_value).value
    except ValueError:
        names = ", ".join(repr(a.value) for a in GroupLoadStoreAlgorithm)
        raise TypedPipelineError(
            f"coop.{kind}: unknown algorithm {ty.literal_value!r}; "
            f"use one of {names}"
        ) from None


_COMMON_ARRAY_PAYLOAD = {
    "load": (
        "cuda.coop.load requires output to be a fixed-size ThreadData "
        "payload in the common API; use cuda.coop.numba_mlir for "
        "backend-qualified local-array payload support"
    ),
    "store": (
        "cuda.coop.store accepts only a scalar or fixed-size ThreadData "
        "value payload in the common API; use cuda.coop.numba_mlir for "
        "backend-qualified local-array payload support"
    ),
}


def _check_payload(kind: str, payload: Any, qualified: bool) -> None:
    if isinstance(payload, ThreadDataType):
        return
    if isinstance(payload, nb.types.Array):
        if qualified:
            return
        raise TypedPipelineError(_COMMON_ARRAY_PAYLOAD[kind])
    raise TypedPipelineError(
        f"coop.{kind}: {_PAYLOAD[kind]} must be a coop.ThreadData, got "
        f"{payload}"
    )


def _bind_operation(kind: str, args, kws) -> dict:
    bound = dict(zip(("group", _MEMORY[kind], _PAYLOAD[kind]), args))
    bound.update(kws)
    return bound


def is_scalar_value(ty: Any) -> bool:
    """Whether ``ty`` is a scalar ``store`` value rather than a payload."""

    return isinstance(nb.unliteral(ty), (nb.types.Number, nb.types.Boolean))


_PENDING_EXTENT = object()


def _request(
    kind: str, args, kws, *, qualified: bool = False, extent: int | None = None
) -> Any:
    """Validate a typed call and return its interned request.

    The request has no controls: whether a control is static or runtime
    depends on where its value comes from, which site analysis reads
    from the IR. A scalar ``store`` value stores one item per thread of the
    destination's dtype; site analysis checks the value against it.

    Return ``None`` while the payload's dtype is still undefined, so that a
    later typing round retries with the refined payload. A qualified call
    with a plain-array payload has no extent in its types; it returns
    ``_PENDING_EXTENT`` until site analysis supplies ``extent``.
    """

    from ..._core.dtype_policy import validate_common_numeric_dtype_name

    bound = _bind_operation(kind, args, kws)
    storage = tuple(storage_options(bound.get("temp_storage")).items())
    group = bound.get("group")
    if not isinstance(group, CoopGroupType):
        raise TypedPipelineError(
            f"coop.{kind}: expected a cooperative thread group, got {group}"
        )
    if not group.is_block and storage:
        raise TypedPipelineError(
            f"cuda.coop.numba_mlir.{kind} temp_storage is not supported for "
            "Warp groups; omit it so the implementation can provide "
            "per-group storage"
        )
    memory = bound.get(_MEMORY[kind])
    if not isinstance(memory, nb.types.Array):
        raise TypedPipelineError(
            f"coop.{kind}: {_MEMORY[kind]} must be an array, got {memory}"
        )
    payload = bound.get(_PAYLOAD[kind])
    if kind == "store" and is_scalar_value(payload):
        dtype, items = memory.dtype, 1
    else:
        _check_payload(kind, payload, qualified)
        assert isinstance(payload, nb.types.Array)
        if not payload.dtype.is_precise():
            return None
        if payload.dtype != memory.dtype:
            raise TypedPipelineError(
                f"coop.{kind}: payload dtype {payload.dtype} conflicts with "
                f"{_MEMORY[kind]} dtype {memory.dtype}"
            )
        dtype = payload.dtype
        items = payload.items if isinstance(payload, ThreadDataType) else extent
    algorithm = _algorithm(kind, bound.get("algorithm"))
    if algorithm is None:
        return None
    try:
        validate_common_numeric_dtype_name(str(dtype), operation=kind)
    except TypeError as exc:
        raise TypedPipelineError(str(exc)) from None
    if items is None:
        return _PENDING_EXTENT
    return _intern(kind, group, dtype, items, algorithm, storage=storage)


def request_for(
    kind: str, args, kws, *, qualified: bool = False, extent: int | None = None
) -> LoadStoreRequest:
    """Return the request a typed call built, from its argument types.

    ``extent`` is the extent of a plain-array payload, which its type does
    not carry.
    """

    request = _request(kind, args, kws, qualified=qualified, extent=extent)
    if request is None or request is _PENDING_EXTENT:
        raise AssertionError(f"coop.{kind} reached call resolution untyped")
    return request


def _operation(kind: str, qualified: bool) -> Callable:
    def generic(self, args, kws):
        if _request(kind, args, kws, qualified=qualified) is None:
            return None
        return nb.signature(nb.types.none, *args, *kws.values())

    return generic


# Inline wrappers --------------------------------------------------------------

WRAPPER_SOURCE: dict[str, str] = {}
_WRAPPER_FILENAME = "<cuda.coop typed-pipeline wrapper for {}>"


def is_wrapper_location(loc: Any) -> bool:
    """Whether ``loc`` is inside an inlined wrapper, not the kernel."""

    prefix = _WRAPPER_FILENAME.split("{", 1)[0]
    return str(getattr(loc, "filename", "")).startswith(prefix)


def _parameter_text(p: inspect.Parameter) -> str:
    if p.default is p.empty:
        return p.name
    if eval(repr(p.default), {}) != p.default:
        raise TypeError(f"default of {p.name!r} has no literal spelling")
    return f"{p.name}={p.default!r}"


def make_wrapper(
    public: Callable, qualified_name: str, operation: str
) -> Callable:
    """Generate the inline wrapper of a public ``load`` or ``store`` object.

    The wrapper has the public object's parameters, with keyword-only ones
    relaxed to ordinary parameters: Numba's inliner cannot bind keyword-only
    parameters. The private operation still receives them by
    keyword.
    """

    params = list(inspect.signature(public).parameters.values())
    positional_only = [p for p in params if p.kind is p.POSITIONAL_ONLY]
    ordinary = [
        p for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    header = [_parameter_text(p) for p in positional_only]
    if positional_only:
        header.append("/")
    header += [_parameter_text(p) for p in ordinary]
    call = [p.name for p in positional_only]
    call += [
        p.name if p.kind is p.POSITIONAL_OR_KEYWORD else f"{p.name}={p.name}"
        for p in ordinary
    ]
    if qualified_name.startswith(_QUALIFIED_PREFIX):
        load, store = _coop_qualified_load_op, _coop_qualified_store_op
    else:
        load, store = _coop_load_op, _coop_store_op
    op = load if operation == "load" else store
    name = f"_coop_{operation}_wrapper"
    source = (
        f"def {name}({', '.join(header)}):\n"
        f"    {op.__name__}({', '.join(call)})\n"
    )
    filename = _WRAPPER_FILENAME.format(qualified_name)
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = {op.__name__: op}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    WRAPPER_SOURCE[qualified_name] = source
    return namespace[name]


def result_wrapper(public, qualified_name, operation):
    """Inline a public signature into a private value-returning operation."""

    params = list(inspect.signature(public).parameters.values())
    positional = [p for p in params if p.kind is p.POSITIONAL_ONLY]
    ordinary = [
        p for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
    ]
    header = [_parameter_text(p) for p in positional]
    if positional:
        header.append("/")
    header.extend(_parameter_text(p) for p in ordinary)
    arguments = [p.name for p in positional]
    arguments.extend(f"{p.name}={p.name}" for p in ordinary)
    source = (
        f"def wrapper({', '.join(header)}):\n"
        f"    return operation({', '.join(arguments)})\n"
    )
    filename = _WRAPPER_FILENAME.format(qualified_name)
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = {"operation": operation}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102
    return namespace["wrapper"]


# Registration ----------------------------------------------------------------


_registered = False


def register() -> None:
    """Register a template for every kernel-callable public object.

    Group and storage descriptors register separately. ``ThreadData``,
    ``load`` and ``store`` of both APIs are typed here.
    """

    global _registered
    if _registered:
        return
    from ._inference import numpy_integer_literal, operation_literal_arguments

    for obj, qualified_name in kernel_primitives().items():
        short = qualified_name.rsplit(".", 1)[1]
        name = qualified_name.replace(".", "_")
        operation = group_operation(obj)
        if short in {
            "this_thread",
            "this_warp",
            "this_block",
            "this_cluster",
            "this_grid",
            "TempStorage",
        }:
            continue  # Group typing and lowering are registered together.
        elif short == "ThreadData":
            operation_literal_arguments[obj] = {
                parameter: numpy_integer_literal
                for parameter in (0, "items_per_thread", "alignment")
            }
            register_template(
                obj, _thread_data(qualified_name), name, prefer_literal=True
            )
        elif operation in {"load", "store"}:
            nb.attach_inline_wrapper(
                obj, make_wrapper(obj, qualified_name, operation)
            )
            register_template(obj, _not_inlined(qualified_name), name)
        elif operation in {
            "exchange",
            "shuffle",
            "reduce",
            "sum",
            "scan",
            "inclusive_scan",
            "exclusive_scan",
            "inclusive_sum",
            "exclusive_sum",
        }:
            continue  # Result families register their templates separately.
        else:
            register_template(obj, _rejecting(qualified_name), name)

    for op, kind in tuple(OPERATIONS.items()):
        if kind not in {"load", "store"}:
            continue
        qualified = op in QUALIFIED_OPERATIONS
        register_template(
            op,
            _operation(kind, qualified),
            f"private_{'qualified_' if qualified else ''}{kind}",
            prefer_literal=True,
        )
    from ._inference import OperationInference, operation_inference_contracts

    for op, kind in OPERATIONS.items():
        if kind == "load":
            operation_inference_contracts[op] = OperationInference(
                producers=((2, 1),)
            )
        elif kind == "store":
            operation_inference_contracts[op] = OperationInference(
                stores=((2, 1),)
            )
    nb.typing_registry.register_attr(_ThreadDataAttributes)
    _registered = True


__all__ = [
    "CONTROLS",
    "OPERATIONS",
    "QUALIFIED_OPERATIONS",
    "RESULT_EXTENT_SOURCES",
    "WRAPPER_SOURCE",
    "LoadStoreRequest",
    "constructor_alignment",
    "group_operation",
    "is_scalar_value",
    "is_wrapper_location",
    "kernel_primitives",
    "make_wrapper",
    "register",
    "register_template",
    "request_for",
    "result_wrapper",
    "with_controls",
]
