# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""GDB pretty printer for cuda::std::complex and cuda::complex."""

from __future__ import annotations

import struct
from collections.abc import Iterator
from types import ModuleType

import cccl_common

import gdb
import gdb.printing

_COMPLEX_NAMES = frozenset({"cuda::complex", "cuda::std::complex"})
_PACKED_BITS_FIELD = "__x"


def decode_packed_half(bits: int, type_name: str) -> float:
    """Decode the 16-bit pattern of a __half or __nv_bfloat16 into a float.

    A packed complex stores its parts as CUDA 16-bit floating-point types whose
    only member is the raw bit pattern, so the debugger would otherwise show an
    integer. Every __half and __nv_bfloat16 value is exactly representable as a
    32-bit float: a __nv_bfloat16 is the upper half of one, and a __half is IEEE
    binary16.
    """
    bits &= 0xFFFF
    if "bfloat16" in type_name:
        return struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def _as_float(part: gdb.Value) -> gdb.Value:
    """Return a packed __half / __nv_bfloat16 part as a float gdb.Value."""
    part_type = cccl_common.canonical_type(part.type)
    try:
        bits = int(part[_PACKED_BITS_FIELD])
    except gdb.error:
        return part
    decoded = decode_packed_half(bits, str(part_type))
    return gdb.Value(decoded).cast(gdb.lookup_type("float"))


def _is_cuda_complex(value_type: gdb.Type) -> bool:
    value_type = cccl_common.canonical_type(value_type)
    template_name = cccl_common.template_name(cccl_common.public_type_name(value_type))
    return template_name in _COMPLEX_NAMES


class ComplexPrinter:
    """Expose cuda::std::complex and cuda::complex parts to GDB."""

    def __init__(self, value: gdb.Value) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value
        self.type = cccl_common.canonical_type(value.type)
        self.type_name = cccl_common.public_type_name(self.type)

    def children(self) -> Iterator[tuple[str, gdb.Value]]:
        try:
            real = self.value["__re_"]
            imag = self.value["__im_"]
        except gdb.error:
            packed = self.value["__repr_"]
            real = _as_float(packed["x"])
            imag = _as_float(packed["y"])
        yield "real", real
        yield "imag", imag

    def to_string(self) -> str:
        return self.type_name


class ComplexPrinterLookup(gdb.printing.PrettyPrinter):
    """Select the complex printer by its public class name."""

    def __init__(self) -> None:
        super().__init__("cuda::complex")

    def __call__(self, value: gdb.Value) -> ComplexPrinter | None:
        if _is_cuda_complex(value.type):
            return ComplexPrinter(value)
        return None


def register(objfile: ModuleType) -> None:
    """Register the cuda complex printer with GDB."""
    gdb.printing.register_pretty_printer(objfile, ComplexPrinterLookup(), replace=True)
