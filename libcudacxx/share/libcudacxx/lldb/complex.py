# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::std::complex and cuda::complex."""

from __future__ import annotations

import re
import struct

import cccl_common

import lldb

_COMPLEX_PATTERN = re.compile(r"^cuda::(?:std::)?complex<.+>$")
_CHILD_NAMES = ("real", "imag")
_PACKED_BITS_FIELD = "__x"
InternalDict = dict[str, object]


def _raw_child(value: lldb.SBValue, name: str) -> lldb.SBValue:
    child = value.GetChildMemberWithName(name)
    if child.IsValid():
        return child.GetNonSyntheticValue()
    return child


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


def _as_float(part: lldb.SBValue, name: str) -> lldb.SBValue:
    """Return a packed __half / __nv_bfloat16 part as a float child."""
    raw = _raw_child(part, _PACKED_BITS_FIELD)
    if not raw.IsValid():
        return part.Clone(name)
    decoded = decode_packed_half(
        raw.GetValueAsUnsigned(), cccl_common.canonical_type_name(part.GetType())
    )
    float_bits = struct.unpack("<I", struct.pack("<f", decoded))[0]
    target = part.GetTarget()
    data = lldb.SBData.CreateDataFromUInt32Array(
        target.GetByteOrder(), target.GetAddressByteSize(), [float_bits]
    )
    return part.CreateValueFromData(
        name, data, target.GetBasicType(lldb.eBasicTypeFloat)
    )


def is_cuda_complex(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = cccl_common.canonical_type_name(value_type)
    return _COMPLEX_PATTERN.fullmatch(type_name) is not None


class ComplexSyntheticProvider:
    """Expose complex real and imaginary parts as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.update()

    def update(self) -> bool:
        self.type_name = (
            self.value.GetType()
            .GetCanonicalType()
            .GetUnqualifiedType()
            .GetDisplayTypeName()
            or ""
        )
        self.parts: list[lldb.SBValue] = []
        real = _raw_child(self.value, "__re_")
        imag = _raw_child(self.value, "__im_")
        if real.IsValid() and imag.IsValid():
            self.parts = [real.Clone("real"), imag.Clone("imag")]
            return True
        packed = _raw_child(self.value, "__repr_")
        if not packed.IsValid():
            return False
        real = _raw_child(packed, "x")
        imag = _raw_child(packed, "y")
        if not real.IsValid() or not imag.IsValid():
            return False
        self.parts = [_as_float(real, "real"), _as_float(imag, "imag")]
        return True

    def num_children(self) -> int:
        return len(self.parts)

    def has_children(self) -> bool:
        return bool(self.parts)

    def get_type_name(self) -> str:
        return self.type_name

    def get_child_index(self, name: str) -> int:
        if name in _CHILD_NAMES:
            return _CHILD_NAMES.index(name)
        return -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if 0 <= index < len(self.parts):
            return self.parts[index]
        return None


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda complex formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.ComplexSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_complex"
    )
