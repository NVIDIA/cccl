# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Byte serialization of CCCL descriptors (``cccl_op_t``, ``cccl_iterator_t``, etc.).
"""

from __future__ import annotations

import struct

import numpy as np

from cuda.cccl import __version__ as _PKG_VERSION

from .._bindings import Iterator, IteratorKind, Op, OpKind, TypeEnum, TypeInfo, Value
from .._device_code import DeviceCode

# An opaque 8-byte marker identifying a cuda.compute serialization blob.
_MAGIC = b"CCCLPYS1"


class Writer:
    """Append-only little-endian byte buffer."""

    def __init__(self) -> None:
        self.buf = bytearray()

    def u8(self, v: int) -> None:
        self.buf += struct.pack("<B", v)

    def u32(self, v: int) -> None:
        self.buf += struct.pack("<I", v)

    def u64(self, v: int) -> None:
        self.buf += struct.pack("<Q", v)

    def blob(self, b: bytes) -> None:
        self.u64(len(b))
        self.buf += b

    def text(self, s: str) -> None:
        self.blob(s.encode("utf-8"))

    def getvalue(self) -> bytes:
        return bytes(self.buf)


class Reader:
    """Bounds-checked little-endian reader over a bytes blob."""

    def __init__(self, data: bytes) -> None:
        self._data = memoryview(data)
        self.pos = 0

    def _take(self, n: int) -> memoryview:
        end = self.pos + n
        if end > len(self._data):
            raise ValueError("serialization descriptor blob truncated")
        out = self._data[self.pos : end]
        self.pos = end
        return out

    def u8(self) -> int:
        return struct.unpack("<B", self._take(1))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._take(8))[0]

    def blob(self) -> bytes:
        return bytes(self._take(self.u64()))

    def text(self) -> str:
        return self.blob().decode("utf-8")

    def remaining(self) -> bytes:
        """Bytes after the descriptor region — i.e. the C build_result blob."""
        return bytes(self._data[self.pos :])


# --- framing -----------------------------------------------------------------


def _check_header(r: Reader) -> None:
    """Validate the magic and package-version stamp at the start of a blob."""
    if bytes(r._take(len(_MAGIC))) != _MAGIC:
        raise ValueError(
            "serialization blob: bad magic (not a cuda.compute serialization blob)"
        )
    version = r.text()
    if version != _PKG_VERSION:
        raise ValueError(
            "serialization blob: cuda-cccl version mismatch "
            f"(blob={version!r}, current={_PKG_VERSION!r}); "
            "re-serialize with this version of cuda-cccl"
        )


def begin(algo_tag: str) -> Writer:
    """Start a descriptor sidecar with the magic/version/algo header.

    ``algo_tag`` is the algorithm class's ``__qualname__`` (e.g. ``"_Reduce"``).
    """
    w = Writer()
    w.buf += _MAGIC
    w.text(_PKG_VERSION)
    w.text(algo_tag)
    return w


def open(blob: bytes, expected_algo: str) -> Reader:
    """Validate the header and return a reader positioned at the first field."""
    r = Reader(blob)
    _check_header(r)
    algo = r.text()
    if algo != expected_algo:
        raise ValueError(
            f"serialization blob: wrong algorithm (blob tag={algo!r}, expected={expected_algo!r})"
        )
    return r


def peek_algo(blob: bytes) -> str:
    """Return the algorithm tag (class ``__qualname__``) from a blob header
    without consuming the blob.

    Validates magic + version. Used by the generic ``deserialize`` dispatcher to
    pick the right algorithm reconstructor.
    """
    r = Reader(blob)
    _check_header(r)
    return r.text()


# --- descriptor (de)serialization --------------------------------------------


def write_type_info(w: Writer, ti: TypeInfo) -> None:
    w.u64(ti.size)
    w.u64(ti.alignment)
    w.u32(int(ti.typenum))


def read_type_info(r: Reader) -> TypeInfo:
    size = r.u64()
    alignment = r.u64()
    type_enum = r.u32()
    return TypeInfo(size, alignment, TypeEnum(type_enum))


def write_op(w: Writer, op: Op) -> None:
    # The operator's device code is serialized in full (this is exactly what a
    # normal __call__ passes to execute), so reconstruction needs no JIT. Only
    # per-call op state is omitted.
    w.u32(int(op.operator_type))
    w.text(op.name)
    w.blob(op.ltoir)
    w.text(op.code.kind)
    w.u32(op.state_alignment)
    # State *bytes* are per-call, but the state *size* is structural and fixes
    # op_data.size at construction (the per-call state setter does not update it).
    w.u64(len(op.state))
    extras = op.extra_code
    w.u32(len(extras))
    for dc in extras:
        w.blob(dc.op_bytes)
        w.text(dc.kind)


def read_op(r: Reader) -> Op:
    operator_type = OpKind(r.u32())
    name = r.text()
    code = r.blob()
    code_kind = r.text()
    state_alignment = r.u32()
    state_size = r.u64()
    n_extra = r.u32()
    extras = [DeviceCode(op_bytes=r.blob(), kind=r.text()) for _ in range(n_extra)]
    return Op(
        name=name,
        operator_type=operator_type,
        ltoir=DeviceCode(op_bytes=code, kind=code_kind),
        state=bytes(state_size),  # zero placeholder; real bytes bound per-call
        state_alignment=state_alignment,
        extra_ltoirs=extras,
    )


def write_iterator(w: Writer, it: Iterator) -> None:
    w.u8(1 if it.is_kind_pointer() else 0)
    w.u32(it.alignment)
    write_type_info(w, it.value_type)
    write_op(w, it.advance_op)
    write_op(w, it.dereference_or_assign_op)


def read_iterator(r: Reader) -> Iterator:
    kind = IteratorKind.POINTER if r.u8() else IteratorKind.ITERATOR
    alignment = r.u32()
    value_type = read_type_info(r)
    advance = read_op(r)
    deref = read_op(r)
    # state is bound per-call (set_cccl_iterator_state); start with none.
    return Iterator(alignment, kind, advance, deref, value_type, state=None)


def write_value(w: Writer, val: Value) -> None:
    # Only the type is static; the value bytes are bound per-call.
    write_type_info(w, val.type)


def read_value(r: Reader) -> Value:
    value_type = read_type_info(r)
    # Placeholder state sized to the value type; __call__ rebinds the real bytes.
    placeholder = np.zeros(value_type.size, dtype=np.uint8)
    return Value(value_type, placeholder)
