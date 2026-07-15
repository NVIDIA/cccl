# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Schema-driven serialize/deserialize base for cuda.compute algorithms.

A built-algorithm class declares a ``__serialization_schema__`` listing its
serialized members as ``(attr_name, kind)`` pairs, including its ``build_result``
as a ``BUILD_RESULT(<type>)`` member. ``Serializable`` provides generic
``serialize``/``deserialize`` that walk the schema, so subclasses need no
hand-written codec and both directions share one field order. Subclasses
auto-register by their ``__qualname__`` for the free-function
``deserialize`` dispatcher.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from . import codec


class _Kind:
    """A serialized member kind: writes/reads one value.

    ``obj`` is the instance being (de)serialized; most kinds ignore it, but
    ``CONDITIONAL`` uses it to read a selector member deserialized earlier.
    """

    __slots__ = ()

    def write(self, w: codec.Writer, value: Any, obj: Any) -> None:
        raise NotImplementedError

    def read(self, r: codec.Reader, obj: Any) -> Any:
        raise NotImplementedError


class _Descriptor(_Kind):
    """Iterator / Op / Value descriptor, delegating to the codec codec."""

    __slots__ = ("_write", "_read")

    def __init__(self, writer: Callable, reader: Callable) -> None:
        self._write = writer
        self._read = reader

    def write(self, w: codec.Writer, value: Any, obj: Any) -> None:
        self._write(w, value)

    def read(self, r: codec.Reader, obj: Any) -> Any:
        return self._read(r)


ITER = _Descriptor(codec.write_iterator, codec.read_iterator)
OP = _Descriptor(codec.write_op, codec.read_op)
VALUE = _Descriptor(codec.write_value, codec.read_value)


class _Scalar(_Kind):
    """A fixed-width little-endian unsigned integer (u8/u32/u64)."""

    __slots__ = ("width",)

    def __init__(self, width: int) -> None:
        self.width = width

    def write(self, w: codec.Writer, value: Any, obj: Any) -> None:
        {1: w.u8, 4: w.u32, 8: w.u64}[self.width](int(value))

    def read(self, r: codec.Reader, obj: Any) -> int:
        return {1: r.u8, 4: r.u32, 8: r.u64}[self.width]()


U8, U32, U64 = _Scalar(1), _Scalar(4), _Scalar(8)


class _Bool(_Kind):
    """A boolean, stored as a u8 (0/1)."""

    __slots__ = ()

    def write(self, w: codec.Writer, value: Any, obj: Any) -> None:
        w.u8(1 if value else 0)

    def read(self, r: codec.Reader, obj: Any) -> bool:
        return bool(r.u8())


BOOL = _Bool()


class _Enum(_Kind):
    """An IntEnum member, stored as a u8 and reconstructed as the enum type."""

    __slots__ = ("enum_cls",)

    def __init__(self, enum_cls: Any) -> None:
        self.enum_cls = enum_cls

    def write(self, w: codec.Writer, value: Any, obj: Any) -> None:
        w.u8(int(value))

    def read(self, r: codec.Reader, obj: Any) -> Any:
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

    def write(self, w: codec.Writer, value: Any, obj: Any) -> None:
        w.blob(value.serialize())

    def read(self, r: codec.Reader, obj: Any) -> Any:
        return self.cls.deserialize(r.blob())


def BUILD_RESULT(cls: type) -> _SubObject:
    """Schema kind for an algorithm's ``Device<Algo>BuildResult`` member."""
    return _SubObject(cls)


class _BuildResults(_Kind):
    """A ``{cc: Device<Algo>BuildResult}`` mapping — one compiled build result per
    target compute capability.

    Wire form: ``u32`` count, then for each entry a ``u32`` cc key
    (``cc_major * 10 + cc_minor``) followed by the length-prefixed build_result
    blob. Entries are written in sorted-key order so the encoding is
    deterministic. On read, each build_result is deserialized *without* loading
    (``load=False``); the matching build result is loaded lazily on first call, so a
    multi-arch artifact stays portable across GPUs and needs no live device to
    deserialize.
    """

    __slots__ = ("cls",)

    def __init__(self, cls: Any) -> None:
        self.cls = cls

    def write(self, w: codec.Writer, value: Any, obj: Any) -> None:
        from .._caching import _PerCCBuildResults

        # Wrappers always hold a _PerCCBuildResults (build_for_ccs and
        # read() below both produce one). serialize_build_result takes the
        # per-cc source lock, so serialization cannot observe a source whose
        # first device load is still in progress; a plain dict here would
        # silently bypass that lock.
        assert isinstance(value, _PerCCBuildResults)
        ccs = sorted(value)
        w.u32(len(ccs))
        for cc in ccs:
            w.u32(int(cc))
            w.blob(value.serialize_build_result(cc))

    def read(self, r: codec.Reader, obj: Any) -> Any:
        count = r.u32()
        # A single-target blob must match this device, so validate its cc-major
        # eagerly (clear error at deserialize). A multi-arch blob legitimately
        # carries build results for other archs, so defer the cc check — resolve_build_result
        # picks the matching one at call time. Kernel load stays lazy either way.
        check_cc = count == 1
        entries = [(r.u32(), r.blob()) for _ in range(count)]
        result: dict[int, Any] = {}
        for cc, blob in entries:
            if cc in result:
                raise ValueError(
                    f"duplicate compute-capability key {cc} in build_results blob"
                )
            result[cc] = self.cls.deserialize(blob, load=False, check_cc=check_cc)

        from .._caching import _PerCCBuildResults

        return _PerCCBuildResults(result)


def BUILD_RESULTS(cls: type) -> _BuildResults:
    """Schema kind for a ``{cc: Device<Algo>BuildResult}`` build result mapping."""
    return _BuildResults(cls)


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

    def write(self, w: codec.Writer, value: Any, obj: Any) -> None:
        kind = self._kind(obj)
        if kind is not None:
            kind.write(w, value, obj)

    def read(self, r: codec.Reader, obj: Any) -> Any:
        kind = self._kind(obj)
        return None if kind is None else kind.read(r, obj)


def CONDITIONAL(selector: str, branches: dict) -> _Conditional:
    """Schema kind for a member whose kind is chosen by ``selector``'s value."""
    return _Conditional(selector, branches)


_S = TypeVar("_S", bound="Serializable")


class Serializable:
    """Mixin providing schema-driven serialize/deserialize + registration."""

    __slots__ = ()

    # __qualname__ -> subclass, populated as algorithm modules are imported.
    _registry: dict[str, type[Serializable]] = {}

    # Subclasses declare their serialized members here.
    __serialization_schema__: tuple = ()

    # Construction-time binding of a default-build wrapper's loaded result
    # (see cache_build_results). Annotation only: storage comes from each
    # subclass's __slots__; __init__ or deserialize() below assigns it.
    _bound_build_result: Any

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        Serializable._registry[cls.__qualname__] = cls

    def _after_deserialize(self) -> None:
        """Hook to bind derived, non-serialized state after schema members are read.

        Called once at the end of ``deserialize``. Subclasses that keep a cached
        attribute derived from serialized members (and set it in ``__init__``)
        override this to rebind it; the default is a no-op.
        """

    def serialize(self) -> bytes:
        """Serialize this built algorithm to a self-contained serialization blob."""
        w = codec.begin(type(self).__qualname__)
        for attr, kind in self.__serialization_schema__:
            kind.write(w, getattr(self, attr), self)
        return w.getvalue()

    @classmethod
    def deserialize(cls: type[_S], blob: bytes) -> _S:
        """Reconstruct a built algorithm from a blob; no objects required.

        Members are read in schema order and set on the instance as they are
        read, so a ``CONDITIONAL`` member can consult a selector read earlier.
        """
        r = codec.open(blob, cls.__qualname__)
        obj = cls.__new__(cls)
        # deserialize() bypasses __init__, which is where default-build
        # wrappers bind their loaded result (see cache_build_results); an
        # unbound wrapper resolves per call instead.
        obj._bound_build_result = None
        for attr, kind in cls.__serialization_schema__:
            setattr(obj, attr, kind.read(r, obj))
        obj._after_deserialize()
        return obj
