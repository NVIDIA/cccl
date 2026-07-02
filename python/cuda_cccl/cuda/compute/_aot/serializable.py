# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Schema-driven AoT (de)serialization base for cuda.compute algorithms.

A built-algorithm class declares a ``tag`` (stable wire id) and a
``__serde_schema__`` listing its serialized members as ``(attr_name, kind)``
pairs — including its ``build_result`` as a ``BUILD_RESULT(<type>)`` member.
``Serializable`` then provides generic ``serialize``/``deserialize`` that walk
the schema, so subclasses need no hand-written codec and the two directions
share one field order (they can't drift). Subclasses auto-register by ``tag``
for the free-function ``deserialize`` dispatcher.

Derived state (e.g. a compute-fn chosen from ``build_result``) is *not* in the
schema — express it as a ``@property`` so it needs no post-load step. Fields
used only at build time (never on the load path) are simply omitted.

Irregular algorithms (a member whose kind depends on another member's value)
still subclass ``Serializable`` for the tag/registry/header framing but override
``serialize``/``deserialize``.
"""

from __future__ import annotations

import enum
from typing import Any, Callable, TypeVar

from . import serde


class AlgoTag(enum.IntEnum):
    """Stable wire ids for AoT blobs, stamped into each blob header.

    Values 1-11 mirror the C ``cccl_aot_algo_t``; 12 is reserved for the C
    ``FOR`` (not exposed in Python); 13 is Python-only (transform splits into
    unary/binary, which the single C ``TRANSFORM`` tag can't distinguish).
    Never renumber or reuse a value — old blobs encode it.
    """

    REDUCE = 1
    SCAN = 2
    SEGMENTED_REDUCE = 3
    UNARY_TRANSFORM = 4
    BINARY_SEARCH = 5
    MERGE_SORT = 6
    RADIX_SORT = 7
    SEGMENTED_SORT = 8
    THREE_WAY_PARTITION = 9
    UNIQUE_BY_KEY = 10
    HISTOGRAM = 11
    # 12 reserved for the C FOR algorithm (no Python wrapper)
    BINARY_TRANSFORM = 13
    SELECT = 14


class _Kind:
    """A serialized member kind: writes/reads one value.

    ``obj`` is the instance being (de)serialized; most kinds ignore it, but
    ``CONDITIONAL`` uses it to read a selector member deserialized earlier.
    """

    __slots__ = ()

    def write(self, w: serde.Writer, value: Any, obj: Any) -> None:
        raise NotImplementedError

    def read(self, r: serde.Reader, obj: Any) -> Any:
        raise NotImplementedError


class _Descriptor(_Kind):
    """Iterator / Op / Value descriptor, delegating to the serde codec."""

    __slots__ = ("_write", "_read")

    def __init__(self, writer: Callable, reader: Callable) -> None:
        self._write = writer
        self._read = reader

    def write(self, w: serde.Writer, value: Any, obj: Any) -> None:
        self._write(w, value)

    def read(self, r: serde.Reader, obj: Any) -> Any:
        return self._read(r)


ITER = _Descriptor(serde.write_iterator, serde.read_iterator)
OP = _Descriptor(serde.write_op, serde.read_op)
VALUE = _Descriptor(serde.write_value, serde.read_value)


class _Scalar(_Kind):
    """A fixed-width little-endian unsigned integer (u8/u32/u64)."""

    __slots__ = ("width",)

    def __init__(self, width: int) -> None:
        self.width = width

    def write(self, w: serde.Writer, value: Any, obj: Any) -> None:
        {1: w.u8, 4: w.u32, 8: w.u64}[self.width](int(value))

    def read(self, r: serde.Reader, obj: Any) -> int:
        return {1: r.u8, 4: r.u32, 8: r.u64}[self.width]()


U8, U32, U64 = _Scalar(1), _Scalar(4), _Scalar(8)


class _Bool(_Kind):
    """A boolean, stored as a u8 (0/1)."""

    __slots__ = ()

    def write(self, w: serde.Writer, value: Any, obj: Any) -> None:
        w.u8(1 if value else 0)

    def read(self, r: serde.Reader, obj: Any) -> bool:
        return bool(r.u8())


BOOL = _Bool()


class _Enum(_Kind):
    """An IntEnum member, stored as a u8 and reconstructed as the enum type."""

    __slots__ = ("enum_cls",)

    def __init__(self, enum_cls: Any) -> None:
        self.enum_cls = enum_cls

    def write(self, w: serde.Writer, value: Any, obj: Any) -> None:
        w.u8(int(value))

    def read(self, r: serde.Reader, obj: Any) -> Any:
        return self.enum_cls(r.u8())


def ENUM(enum_cls: Any) -> _Enum:
    """Schema kind for a u8-backed IntEnum member."""
    return _Enum(enum_cls)


class _SubObject(_Kind):
    """A sub-object with its own ``serialize()``/``deserialize()``, carried as a
    length-prefixed blob. Used for the C ``build_result`` and for a nested
    ``Serializable`` (e.g. select wrapping three_way_partition)."""

    __slots__ = ("cls",)

    def __init__(self, cls: Any) -> None:
        self.cls = cls

    def write(self, w: serde.Writer, value: Any, obj: Any) -> None:
        w.blob(value.serialize())

    def read(self, r: serde.Reader, obj: Any) -> Any:
        return self.cls.deserialize(r.blob())


def BUILD_RESULT(cls: type) -> _SubObject:
    """Schema kind for an algorithm's ``Device<Algo>BuildResult`` member."""
    return _SubObject(cls)


def NESTED(cls: type) -> _SubObject:
    """Schema kind for a nested ``Serializable`` member (its blob is embedded)."""
    return _SubObject(cls)


class _Conditional(_Kind):
    """A member whose kind depends on an earlier member's value.

    ``selector`` names a member deserialized *before* this one; ``branches``
    maps each possible selector value to the kind to use (or ``None`` for an
    absent member that (de)serializes to ``None``).
    """

    __slots__ = ("selector", "branches")

    def __init__(self, selector: str, branches: dict) -> None:
        self.selector = selector
        self.branches = branches

    def _kind(self, obj: Any) -> "_Kind | None":
        return self.branches[getattr(obj, self.selector)]

    def write(self, w: serde.Writer, value: Any, obj: Any) -> None:
        kind = self._kind(obj)
        if kind is not None:
            kind.write(w, value, obj)

    def read(self, r: serde.Reader, obj: Any) -> Any:
        kind = self._kind(obj)
        return None if kind is None else kind.read(r, obj)


def CONDITIONAL(selector: str, branches: dict) -> _Conditional:
    """Schema kind for a member whose kind is chosen by ``selector``'s value."""
    return _Conditional(selector, branches)


_S = TypeVar("_S", bound="Serializable")


class Serializable:
    """Mixin providing schema-driven AoT serialize/deserialize + registration."""

    __slots__ = ()

    # tag -> subclass, populated as algorithm modules are imported.
    _registry: dict[int, type[Serializable]] = {}

    # Subclasses set both as class attributes: `_serde_tag = AlgoTag.<X>` and
    # `__serde_schema__ = (...)`.
    __serde_schema__: tuple = ()
    _serde_tag: AlgoTag

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        tag = cls.__dict__.get("_serde_tag")
        if tag is not None:
            Serializable._registry[tag] = cls

    def serialize(self) -> bytes:
        """Serialize this built algorithm to a self-contained AoT blob."""
        w = serde.begin(self._serde_tag)
        for attr, kind in self.__serde_schema__:
            kind.write(w, getattr(self, attr), self)
        return w.getvalue()

    @classmethod
    def deserialize(cls: type[_S], blob: bytes) -> _S:
        """Reconstruct a built algorithm from a blob — no objects required.

        Members are read in schema order and set on the instance as they are
        read, so a ``CONDITIONAL`` member can consult a selector read earlier.
        """
        r = serde.open(blob, cls._serde_tag)
        obj = cls.__new__(cls)
        for attr, kind in cls.__serde_schema__:
            setattr(obj, attr, kind.read(r, obj))
        return obj
