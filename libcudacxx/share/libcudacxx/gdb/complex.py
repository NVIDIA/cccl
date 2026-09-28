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


def decode_packed_halves(
    x_bits: int, y_bits: int, type_name: str
) -> tuple[float, float]:
    """Decode both 16-bit patterns of a packed complex into a pair of floats.

    A packed complex stores its parts as CUDA 16-bit floating-point types whose
    only member is the raw bit pattern, so the debugger would otherwise show an
    integer. Every __half and __nv_bfloat16 value is exactly representable as a
    32-bit float: a __half is IEEE binary16, and a __nv_bfloat16 is the upper
    half of a binary32 whose lower half the pad bytes below supply.
    """
    if "bfloat16" in type_name:
        return struct.unpack("<2f", struct.pack("<2xH2xH", x_bits, y_bits))
    return struct.unpack("<2e", struct.pack("<2H", x_bits, y_bits))


def _packed_bits(part: gdb.Value) -> int | None:
    """Return the raw bit pattern of a packed part, or None if it has none."""
    part_type = cccl_common.canonical_type(part.type)
    # A packed part is a class type, but fields() raises on anything scalar.
    if part_type.code not in (gdb.TYPE_CODE_STRUCT, gdb.TYPE_CODE_UNION):
        return None
    if not any(field.name == _PACKED_BITS_FIELD for field in part_type.fields()):
        return None
    return int(part[_PACKED_BITS_FIELD])


def _packed_parts(packed: gdb.Value) -> tuple[gdb.Value, gdb.Value]:
    """Return the parts of a packed complex as floats, or unchanged."""
    x = packed["x"]
    y = packed["y"]
    x_bits = _packed_bits(x)
    y_bits = _packed_bits(y)
    if x_bits is None or y_bits is None:
        return x, y
    real, imag = decode_packed_halves(
        x_bits, y_bits, str(cccl_common.canonical_type(x.type))
    )
    float_type = gdb.lookup_type("float")
    return gdb.Value(real).cast(float_type), gdb.Value(imag).cast(float_type)


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
        fields = {field.name for field in self.type.fields()}
        if {"__re_", "__im_"} <= fields:
            real = self.value["__re_"]
            imag = self.value["__im_"]
        else:
            real, imag = _packed_parts(self.value["__repr_"])
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
