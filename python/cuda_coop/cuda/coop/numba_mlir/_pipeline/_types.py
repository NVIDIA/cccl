# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Numba types for cooperative groups and per-thread payloads.

``CoopGroupType`` is the type of ``this_block()``. It carries coop's own
``ThreadGroup`` request and no launch dimensions; call resolution reads the
block from the launch once typing is done.

``ThreadDataType`` is the type of ``ThreadData(items, dtype=...)``. It is a
one-dimensional C-contiguous ``types.Array``, so indexing and every array
operation type as usual, with the extent and the strength of the evidence
that fixed its dtype added to the type key. Strength is what lets typing
apply coop's precedence instead of "first refinement wins":

=============  ========================================================
Strength       Evidence
=============  ========================================================
explicit       ``dtype=`` on the constructor
producer       the payload is the output of a ``load``
store-dst      the payload is the value of a ``store`` (its destination)
write          an indexed write ``td[i] = value``
=============  ========================================================

Stronger evidence replaces weaker evidence at a join; equal evidence that
disagrees is a conflict, which the operation's template reports. A join with
a plain ``types.Array`` is a typing error: the plain array's extent is not
part of its type, so no cooperative operation could be specialized on it.
"""

from __future__ import annotations

from typing import Any

from . import _bridges as nb
from ._errors import TypedPipelineError

WRITE, STORE_DST, PRODUCER, EXPLICIT = 0, 1, 2, 3

STRENGTH_NAMES = {
    EXPLICIT: "explicit",
    PRODUCER: "producer",
    STORE_DST: "store-dst",
    WRITE: "write",
}


class CoopGroupType(nb.types.Type):
    """A cooperative group request, without launch dimensions.

    Parameters
    ----------
    group : cuda.coop._core.thread_group.ThreadGroup
        The request coop's own ``this_block()`` returns on the host.
    """

    def __init__(self, group: Any) -> None:
        self.group = group
        suffix = group.symbol_suffix.removesuffix("_current")
        super().__init__(name=f"coop.group[{suffix}]")

    @property
    def key(self) -> tuple:
        return ("coop.group", self.group.semantic_key)

    @property
    def is_block(self) -> bool:
        return self.group.kind == "block"


class ThreadDataType(nb.types.Array):
    """A per-thread payload: an array whose extent is part of its type.

    Parameters
    ----------
    dtype : numba type
        Element type, or ``types.undefined`` until evidence arrives.
    items : int
        Items per thread.
    strength : int
        Strength of the evidence that fixed ``dtype``; ignored while the
        dtype is undefined.
    alignment : int or None
        The constructor's ``alignment=``. It is an attribute of the
        allocation, not of the value, so it is not part of the key.
    """

    def __init__(
        self,
        dtype: Any,
        items: int,
        strength: int = EXPLICIT,
        alignment: int | None = None,
        *,
        origins: frozenset = frozenset(),
        inferred_origins: frozenset = frozenset(),
    ) -> None:
        self.items = int(items)
        precise = dtype.is_precise()
        self.strength = strength if precise else WRITE
        self.alignment = alignment
        self.origins = origins
        self.inferred_origins = inferred_origins
        tag = f", {STRENGTH_NAMES[self.strength]}" if precise else ""
        super().__init__(
            dtype,
            1,
            "C",
            name=f"coop.ThreadData({dtype}, items={self.items}{tag})",
        )

    @property
    def key(self) -> tuple:
        return (
            self.dtype,
            self.items,
            self.strength,
            self.origins,
            self.inferred_origins,
        )

    def __repr__(self) -> str:
        return self.name

    def with_dtype(self, dtype: Any, strength: int) -> ThreadDataType:
        return ThreadDataType(
            dtype,
            self.items,
            strength,
            self.alignment,
            origins=self.origins,
            inferred_origins=self.inferred_origins,
        )

    def with_origins(self, origins, inferred_origins) -> ThreadDataType:
        """Retain allocation policy through native aliases and tuple joins."""

        return ThreadDataType(
            self.dtype,
            self.items,
            self.strength,
            self.alignment,
            origins=origins,
            inferred_origins=inferred_origins,
        )

    def refined(self, dtype: Any, strength: int) -> ThreadDataType:
        """Apply new dtype evidence of ``strength`` to this payload.

        Conflicting evidence of equal or lower strength leaves the payload
        unchanged, so that the operation's template reports the conflict.
        """

        if not self.dtype.is_precise():
            return self.with_dtype(dtype, strength)
        if self.dtype == dtype:
            return self.with_dtype(dtype, max(self.strength, strength))
        if strength > self.strength:
            return self.with_dtype(dtype, strength)
        return self

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
        # A dtype-only copy is how Numba refines an array from an indexed
        # write. Any other copy is a view, which is a plain array.
        if ndim is None and layout is None and not readonly:
            if dtype is None:
                return self
            if not self.dtype.is_precise():
                return self.with_dtype(dtype, WRITE)
            return self.with_dtype(dtype, self.strength)
        return nb.types.Array(
            self.dtype if dtype is None else dtype,
            self.ndim if ndim is None else ndim,
            self.layout if layout is None else layout,
            readonly=bool(readonly),
        )

    def unify(self, typingctx, other):
        if isinstance(other, nb.types.Array) and not isinstance(
            other, ThreadDataType
        ):
            raise TypedPipelineError(
                f"coop.ThreadData ({self}) cannot share a variable with a "
                f"plain array ({other}): its extent is not part of the "
                "array's type. Use coop.ThreadData on every path."
            )
        if not isinstance(other, ThreadDataType) or other.items != self.items:
            return None
        merged = self.with_origins(
            self.origins | other.origins,
            self.inferred_origins | other.inferred_origins,
        )
        if not other.dtype.is_precise():
            return merged
        if not self.dtype.is_precise():
            return merged.with_dtype(other.dtype, other.strength)
        if self.dtype == other.dtype:
            return merged.with_dtype(
                self.dtype, max(self.strength, other.strength)
            )
        if self.strength != other.strength:
            win = self if self.strength > other.strength else other
            return merged.with_dtype(win.dtype, win.strength)
        return None

    # Array infers only its own safe conversion; extensions may return exact.
    def can_convert_to(self, typingctx, other) -> nb.Conversion | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        if (
            isinstance(other, ThreadDataType)
            and other.items == self.items
            and (not self.dtype.is_precise() or other.dtype == self.dtype)
        ):
            return nb.Conversion.exact
        return super().can_convert_to(typingctx, other)


class _I8Model(nb.PrimitiveModel):
    """A group value is a placeholder byte; it carries no run-time data."""

    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, nb.IntegerType.get_signless(8))


_registered = False


def register() -> None:
    """Register the data models of the cooperative types once."""

    global _registered
    if _registered:
        return
    nb.register_model(CoopGroupType)(_I8Model)
    nb.register_model(ThreadDataType)(nb.ArrayModel)
    _registered = True


__all__ = [
    "EXPLICIT",
    "PRODUCER",
    "STORE_DST",
    "STRENGTH_NAMES",
    "WRITE",
    "CoopGroupType",
    "ThreadDataType",
    "register",
]
