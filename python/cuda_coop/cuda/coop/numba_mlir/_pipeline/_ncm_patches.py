# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Patch Numba-CUDA-MLIR where the typed pipeline needs a compiler change.

Each patch replaces attributes in Numba-CUDA-MLIR with versions that
behave as an upstream ask would make them, and is listed with that ask in the
``_bridges`` table. The typed pipeline's ``_activate`` applies them once,
before it registers anything; the default pipeline never imports this
module, so it leaves the compiler untouched.

A patch is pinned to what it was written against: Numba-CUDA-MLIR
``NCM_VERSION`` and a digest of the source of every attribute it replaces or
relies on. ``apply`` checks every pin before it changes anything, and raises
``TypedPipelineUnavailableError``, naming the patch and its ask, on the first
mismatch.

Inference patches preserve previously accepted kernels without cooperative
types in two ways, described by each patch:

* error to success: it acts only where the compiler would have refused the
  kernel;
* cooperative types only: it acts only on values whose type is, or is a
  tuple holding, a ``ThreadDataType``.

Launch preparation also runs for ordinary kernels in typed mode. It retains
generic specialization and cache eligibility but adds a ready-overload lookup
so requirements discovered during a cooperative compile reach the first launch.
"""

from __future__ import annotations

import hashlib
import inspect
import linecache
import operator
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, cast

import numba_cuda_mlir

from . import _bridges as nb
from ._types import ThreadDataType

NCM_VERSION = "0.5.4"

try:
    from numba_cuda_mlir import descriptor
    from numba_cuda_mlir.numba_cuda.core import typeinfer
    from numba_cuda_mlir.typing import cuda as cuda_typing
except ImportError as exc:
    raise nb.TypedPipelineUnavailableError(
        "The cuda.coop typed pipeline patches Numba-CUDA-MLIR "
        f"{NCM_VERSION}; the installed compiler is missing {exc.name or exc}."
    ) from exc


@dataclass(frozen=True)
class Pin:
    """An attribute a patch replaces or relies on, as it was written against.

    ``digest`` is the start of the SHA-256 of the attribute's source, or
    ``None`` when the owner must not define the attribute itself.
    """

    owner: Any
    name: str
    digest: str | None


@dataclass(frozen=True)
class Patch:
    """A compiler change, the ask that retires it, and its pins."""

    name: str
    ask: str
    pins: tuple[Pin, ...]
    build: Callable[[], tuple[tuple[Any, str, Any], ...]]


def source_digest(obj: Any) -> str | None:
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def _pinned(pin: Pin) -> bool:
    if pin.digest is None:
        return pin.name not in vars(pin.owner)
    return source_digest(vars(pin.owner).get(pin.name)) == pin.digest


def check(patch: Patch) -> None:
    """Raise unless ``patch`` matches the installed compiler."""

    installed = numba_cuda_mlir.__version__
    if installed != NCM_VERSION:
        changed = f"the installed compiler is {installed}"
    else:
        stale = [p for p in patch.pins if not _pinned(p)]
        if not stale:
            return
        owner = stale[0].owner
        owner_name = getattr(owner, "__qualname__", owner.__name__)
        changed = f"{owner_name}.{stale[0].name} has changed"
    raise nb.TypedPipelineUnavailableError(
        f"The cuda.coop typed pipeline's {patch.name!r} patch was written "
        f"against Numba-CUDA-MLIR {NCM_VERSION}, but {changed}. The patch is "
        f"retired by {patch.ask}; until then, unset "
        "CUDA_COOP_NUMBA_MLIR_PIPELINE to use the default pipeline."
    )


def holds_thread_data(ty: Any) -> bool:
    """Whether ``ty`` is a ``ThreadDataType`` or a tuple holding one."""

    if isinstance(ty, ThreadDataType):
        return True
    return isinstance(ty, nb.types.BaseTuple) and any(
        holds_thread_data(item) for item in cast(Iterable[Any], ty)
    )


# Argument-sized cuda.local.array (U-P-1) ------------------------------------

_LOCAL_ARRAY_PARAMETERS = ("shape", "dtype", "alignment", "alignas")


def _fold_local_array_arguments(args, kws) -> tuple:
    bound = dict(zip(_LOCAL_ARRAY_PARAMETERS, args))
    bound.update(dict(kws))
    return tuple(bound[n] for n in _LOCAL_ARRAY_PARAMETERS if n in bound)


def _local_array_generic(self) -> Callable:
    """Type ``cuda.local.array`` and request a non-literal shape as a literal.

    Numba-CUDA-MLIR finds no match for a ``cuda.local.array`` whose shape is
    a non-literal integer. When the call would type with a literal shape,
    this raises ``ForceLiteralArg`` for the shape instead, so a shape that
    is a kernel argument is retried as a literal: the kernel compiles once
    per distinct value of that argument. A shape that is not a kernel
    argument is still refused, with Numba's "Cannot request literal type".
    Tuple shapes are unchanged, since the launcher retries only scalar
    arguments as literals.

    Scope: error to success. A call that typed before types the same.
    """

    typer = cuda_typing.GenericArrayTemplate.generic(self)

    def literal_shape_typer(shape, dtype, alignment=None, alignas=None):
        sig = typer(shape, dtype, alignment, alignas)
        if (
            sig is None
            and isinstance(shape, nb.types.Integer)
            and not isinstance(shape, nb.types.IntegerLiteral)
            and typer(nb.types.IntegerLiteral(1), dtype, alignment, alignas)
            is not None
        ):
            raise nb.force_literal_arg({0}, _fold_local_array_arguments)
        return sig

    return literal_shape_typer


LOCAL_ARRAY_SHAPE = Patch(
    name="argument-sized cuda.local.array",
    ask="U-P-1 (cuda.local.array requests a literal shape)",
    pins=(
        Pin(cuda_typing.GenericArrayTemplate, "generic", "280ceb177adabb9b"),
        Pin(cuda_typing.LocalArrayTemplate, "generic", None),
    ),
    build=lambda: (
        (cuda_typing.LocalArrayTemplate, "generic", _local_array_generic),
    ),
)


# Payload through a tuple alias (U-H1-4) -------------------------------------


def _tuple_refinement() -> tuple[tuple[type, str, Any], ...]:
    """Refine a tuple's items from a refined projection of the tuple.

    Numba-CUDA-MLIR's ``static_getitem`` and ``build_tuple`` constraints
    have no ``refine`` hook, so refining ``x = (td,)[0]`` never reaches
    ``td``. With this patch a refined projection refines the tuple's item
    type, and a refined tuple refines the variables it was built from.

    Scope: cooperative types only. A constraint takes part only when the
    tuple holds a ``ThreadDataType``, and refines only items that do.
    """

    static_getitem = typeinfer.StaticGetItemConstraint.__call__
    build_tuple = typeinfer.BuildTupleConstraint.__call__

    def static_getitem_call(self, typeinfer_):
        static_getitem(self, typeinfer_)
        tuple_var = typeinfer_.typevars[self.value.name]
        if (
            isinstance(self.index, int)
            and tuple_var.defined
            and holds_thread_data(tuple_var.getone())
        ):
            typeinfer_.refine_map.setdefault(self.target, self)

    def static_getitem_refine(self, typeinfer_, target_type):
        if not holds_thread_data(target_type):
            return
        items = list(typeinfer_.typevars[self.value.name].getone().types)
        items[self.index] = target_type
        typeinfer_.add_type(
            self.value.name,
            nb.types.BaseTuple.from_types(items),
            loc=self.loc,
        )

    def build_tuple_call(self, typeinfer_):
        build_tuple(self, typeinfer_)
        target = typeinfer_.typevars[self.target]
        if target.defined and holds_thread_data(target.getone()):
            typeinfer_.refine_map.setdefault(self.target, self)

    def build_tuple_refine(self, typeinfer_, target_type):
        if not isinstance(target_type, nb.types.BaseTuple):
            return
        for item, item_type in zip(
            self.items, cast(Iterable[Any], target_type)
        ):
            if holds_thread_data(item_type) and item_type.is_precise():
                typeinfer_.add_type(
                    item.name, item_type, loc=self.loc, unless_locked=True
                )

    return (
        (typeinfer.StaticGetItemConstraint, "__call__", static_getitem_call),
        (typeinfer.StaticGetItemConstraint, "refine", static_getitem_refine),
        (typeinfer.BuildTupleConstraint, "__call__", build_tuple_call),
        (typeinfer.BuildTupleConstraint, "refine", build_tuple_refine),
    )


TUPLE_REFINEMENT = Patch(
    name="tuple refinement",
    ask="U-H1-4 (refine hooks for tuples)",
    pins=(
        Pin(typeinfer.StaticGetItemConstraint, "__call__", "8b86dc739bada065"),
        Pin(typeinfer.StaticGetItemConstraint, "refine", None),
        Pin(typeinfer.BuildTupleConstraint, "__call__", "c54497d13b566d89"),
        Pin(typeinfer.BuildTupleConstraint, "refine", None),
    ),
    build=_tuple_refinement,
)

# Phi of two ThreadData constructors (U-H1-3) --------------------------------


class _FanOut:
    """Every assignment that feeds one variable, refined together."""

    def __init__(self, members: list) -> None:
        self.members = members

    def refine(self, typeinfer_, target_type) -> None:
        for member in list(self.members):
            member.refine(typeinfer_, target_type)


def _join_refinement() -> tuple[tuple[type, str, Any], ...]:
    """Refine every input of a phi, and let a constructor's refine pass.

    Numba-CUDA-MLIR keeps one refine hook per variable, so when a phi joins
    two ``ThreadData`` constructors, refining the join refines only the last
    input typed. And a call whose result is refined raises "no type
    refinement implemented" unless it is a getitem; propagation drops the
    error, but it stops the refinement of the remaining inputs. With this
    patch every assignment feeding a ``ThreadData`` variable is refined, and
    refining a ``ThreadData`` call's result does nothing: typing adopts the
    refined type for the call's signature on its next round.

    Scope: cooperative types only. Inputs are kept only for a variable whose
    type holds a ``ThreadDataType``, and only a refinement to such a type
    passes a call.
    """

    propagate = typeinfer.Propagate.__call__
    call_refine = typeinfer.CallConstraint.refine

    def propagate_call(self, typeinfer_):
        existing = typeinfer_.refine_map.get(self.dst)
        propagate(self, typeinfer_)
        target = typeinfer_.typevars[self.dst]
        if (
            existing is None
            or existing is self
            or not target.defined
            or not holds_thread_data(target.getone())
        ):
            return
        if isinstance(existing, _FanOut):
            if self not in existing.members:
                existing.members.append(self)
            typeinfer_.refine_map[self.dst] = existing
        elif isinstance(existing, typeinfer.Propagate):
            typeinfer_.refine_map[self.dst] = _FanOut([existing, self])

    def call_constraint_refine(self, typeinfer_, updated_type):
        if self.func != operator.getitem and holds_thread_data(updated_type):
            return
        call_refine(self, typeinfer_, updated_type)

    return (
        (typeinfer.Propagate, "__call__", propagate_call),
        (typeinfer.CallConstraint, "refine", call_constraint_refine),
    )


JOIN_REFINEMENT = Patch(
    name="phi refinement",
    ask=(
        "U-H1-3 (refine every phi input; a constructor's refine hook does "
        "not raise)"
    ),
    pins=(
        Pin(typeinfer.Propagate, "__call__", "f25d7d19db3d8210"),
        Pin(typeinfer.CallConstraint, "refine", "4b828831aeb90ba5"),
    ),
    build=_join_refinement,
)


def _cooperative_inference() -> tuple[tuple[Any, str, Any], ...]:
    """Construct the native solver with cooperative mutation constraints."""

    from ._inference import CooperativeTypeInferer

    return ((typeinfer, "TypeInferer", CooperativeTypeInferer),)


COOPERATIVE_INFERENCE = Patch(
    name="cooperative inference constraints",
    ask="a supported native TypeInferer factory and mutation-constraint hook",
    pins=(Pin(typeinfer, "TypeInferer", "d61c50b7d4f6f009"),),
    build=_cooperative_inference,
)


def _prepare_typed_launches() -> tuple[tuple[Any, str, Any], ...]:
    """Prepare compiler metadata before the first native kernel launch.

    A typed rewrite can discover dynamic scratch during initial compilation,
    before the dispatcher knows that it needs launch metadata. Use NCM's
    existing preparation, generation, literal-retry and TLS lifecycle for
    every typed-mode launch. Ordinary kernels retain generic specialization
    and disk-cache eligibility, with an additional ready-overload lookup.

    Only the preparation predicate changes. Keeping the pinned implementation
    avoids maintaining a second copy of NCM's launcher and cleanup logic.
    """

    original = descriptor._ArgMarshaller._launch
    source = dedent(inspect.getsource(original))
    predicate = dedent("""\
            if dispatcher is not None and (
                _planner_registry.has_planners
                or getattr(dispatcher, "_requires_launch_config", False)
                or self._launch_config is not None
                or literal_prelaunch
            ):
    """).strip()
    # The method body remains indented after dedenting its definition.
    predicate = "\n".join("        " + line for line in predicate.splitlines())
    if source.count(predicate) != 1:
        raise nb.TypedPipelineUnavailableError(
            "The pinned NCM launch preparation predicate was not found"
        )
    source = source.replace(predicate, "        if dispatcher is not None:")
    filename = "<cuda.coop NCM launch preparation>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = {}
    exec(compile(source, filename, "exec"), original.__globals__, namespace)  # noqa: S102
    replacement = namespace[original.__name__]
    replacement.__qualname__ = original.__qualname__
    return ((descriptor._ArgMarshaller, "_launch", replacement),)


LAUNCH_PREPARATION = Patch(
    name="typed first-launch preparation",
    ask="a compiler-extension prelaunch preparation hook",
    pins=(
        Pin(descriptor._ArgMarshaller, "_launch", "a1eb72b546367971"),
        Pin(
            descriptor.MLIRDispatcher, "_prepare_for_launch", "b770adf406b91764"
        ),
        Pin(descriptor.MLIRDispatcher, "_get_ready_launch", "f80e6fe0a159cd27"),
    ),
    build=_prepare_typed_launches,
)

PATCHES = (
    LOCAL_ARRAY_SHAPE,
    TUPLE_REFINEMENT,
    JOIN_REFINEMENT,
    COOPERATIVE_INFERENCE,
    LAUNCH_PREPARATION,
)

_ABSENT = object()
_replaced: list[tuple[Any, str, Any]] = []


def apply() -> None:
    """Check every patch, then apply them all; later calls do nothing."""

    if _replaced:
        return
    for patch in PATCHES:
        check(patch)
    for patch in PATCHES:
        for owner, name, value in patch.build():
            _replaced.append((owner, name, vars(owner).get(name, _ABSENT)))
            setattr(owner, name, value)


def replaced() -> tuple[tuple[Any, str], ...]:
    """The ``(owner, attribute)`` pairs ``apply`` replaced."""

    return tuple((owner, name) for owner, name, _ in _replaced)


__all__ = [
    "COOPERATIVE_INFERENCE",
    "JOIN_REFINEMENT",
    "LAUNCH_PREPARATION",
    "LOCAL_ARRAY_SHAPE",
    "NCM_VERSION",
    "PATCHES",
    "TUPLE_REFINEMENT",
    "Patch",
    "Pin",
    "apply",
    "check",
    "holds_thread_data",
    "replaced",
    "source_digest",
]
