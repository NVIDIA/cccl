# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printer for cuda::std::mdspan and its cuda:: accessibility wrappers."""

from __future__ import annotations

import re
from typing import NamedTuple

import cccl_common

import lldb

_MDSPAN_PATTERN = re.compile(r"^cuda::std::mdspan<.+>$")
_EBCO_PATTERN = re.compile(r"__mdspan_ebco<")
_EBCO_IMPL_PATTERN = re.compile(r"__mdspan_ebco_impl<\s*(\d+)\s*,")
_EXTENTS_PATTERN = re.compile(r"extents<\s*([^>]*)>$")
_LAYOUT_KINDS = ("layout_left", "layout_right", "layout_stride")
# cudaMemcpyDefault: infer host/device direction from the pointers themselves.
_CUDA_MEMCPY_DEFAULT = 4
# cuda:: accessibility wrappers around cuda::std::mdspan.
_WRAPPER_MDSPAN_NAMES = frozenset(
    {
        "host_mdspan",
        "device_mdspan",
        "managed_mdspan",
        "restrict_mdspan",
        "shared_memory_mdspan",
    }
)
InternalDict = dict[str, object]


def _dynamic_extent(target: lldb.SBTarget) -> int:
    """Return cuda::std::dynamic_extent, using the target's real size_t width."""
    size_t_type = target.FindFirstType("size_t")
    size_t_bits = (size_t_type.GetByteSize() or 8) * 8
    return (1 << size_t_bits) - 1


def is_cuda_mdspan(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    type_name = cccl_common.canonical_type_name(value_type)
    if _MDSPAN_PATTERN.fullmatch(type_name) is not None:
        return True
    template_name = type_name.split("<", 1)[0]
    return (
        template_name.startswith("cuda::")
        and template_name.rsplit("::", 1)[-1] in _WRAPPER_MDSPAN_NAMES
    )


def _direct_bases(sb_type: lldb.SBType) -> list[lldb.SBTypeMember]:
    return [
        sb_type.GetDirectBaseClassAtIndex(i)
        for i in range(sb_type.GetNumberOfDirectBaseClasses())
    ]


def _template_name(sb_type: lldb.SBType) -> str:
    name = sb_type.GetDisplayTypeName() or sb_type.GetName() or ""
    return name.split("<", 1)[0]


def _mdspan_base_type(sb_type: lldb.SBType) -> lldb.SBType | None:
    """Find the real cuda::std::mdspan type, skipping any cuda:: wrapper."""
    current = sb_type
    while _template_name(current).rsplit("::", 1)[-1] != "mdspan":
        bases = _direct_bases(current)
        if len(bases) != 1:
            return None
        current = bases[0].GetType()
    return current


def _find_ebco_base(value: lldb.SBValue) -> lldb.SBValue | None:
    """Descend through transparent wrapper bases to an ``__mdspan_ebco<...>``.

    Recurses through layout_stride's ``__mapping_base<...>`` and cuda:: wrappers like ``host_mdspan``.
    """
    bases = _direct_bases(value.GetType())
    for base in bases:
        if _EBCO_PATTERN.search(base.GetName() or ""):
            return value.CreateChildAtOffset(
                "__mdspan_ebco", base.GetOffsetInBytes(), base.GetType()
            )
    if len(bases) == 1:
        base = bases[0]
        wrapper = value.CreateChildAtOffset(
            "__wrapper", base.GetOffsetInBytes(), base.GetType()
        )
        return _find_ebco_base(wrapper)
    return None


def _ebco_element(
    ebco_value: lldb.SBValue, index: int, name: str
) -> lldb.SBValue | None:
    """Extract element ``index`` from an ``__mdspan_ebco<...>`` value.

    Held in ``__elem_``, or (EBCO) built from freestanding data to dodge
    ``CreateChildAtOffset``'s cache colliding on a shared empty-element offset.
    """
    for base in _direct_bases(ebco_value.GetType()):
        match = _EBCO_IMPL_PATTERN.search(base.GetName() or "")
        if match is None or int(match.group(1)) != index:
            continue
        impl_type = base.GetType()
        if impl_type.GetNumberOfFields() == 0:
            element_type = impl_type.GetDirectBaseClassAtIndex(0).GetType()
            byte_size = element_type.GetByteSize() or 1
            target = ebco_value.GetTarget()
            data = lldb.SBData()
            error = lldb.SBError()
            data.SetData(
                error,
                bytes(byte_size),
                target.GetByteOrder(),
                target.GetAddressByteSize(),
            )
            if error.Fail():
                return None
            return ebco_value.CreateValueFromData(name, data, element_type)
        impl = ebco_value.CreateChildAtOffset(name, base.GetOffsetInBytes(), impl_type)
        return impl.GetChildMemberWithName("__elem_")
    return None


def _descend_unique_base(value: lldb.SBValue, levels: int) -> lldb.SBValue | None:
    """Cast down through ``levels`` of single private-base inheritance."""
    for _ in range(levels):
        bases = _direct_bases(value.GetType())
        if len(bases) != 1:
            return None
        base = bases[0]
        value = value.CreateChildAtOffset(
            "__base", base.GetOffsetInBytes(), base.GetType()
        )
    return value


def _static_extents(extents_type: lldb.SBType) -> list[int] | None:
    """Parse the static extents (dynamic_extent sentinel kept) out of an
    ``extents<IndexType, Values...>`` type's display name."""
    type_name = extents_type.GetDisplayTypeName() or extents_type.GetName() or ""
    match = _EXTENTS_PATTERN.search(type_name)
    if match is None:
        return None
    parts = [part.strip() for part in match.group(1).split(",")]
    parts = [part for part in parts if part]
    if not parts:
        return None
    return [int(value) for value in parts[1:]]


def _dynamic_values(extents_value: lldb.SBValue, count: int) -> list[int] | None:
    if count == 0:
        return []
    array_value = _descend_unique_base(extents_value, 2)
    if array_value is None:
        return None
    vals = array_value.GetChildMemberWithName("__vals_")
    if not vals.IsValid():
        return None
    return [vals.GetChildAtIndex(i).GetValueAsSigned(0) for i in range(count)]


def _combined_extents(
    static_values: list[int], dynamic_values: list[int], dynamic_extent: int
) -> list[int]:
    dynamic_iter = iter(dynamic_values)
    return [
        next(dynamic_iter) if value == dynamic_extent else value
        for value in static_values
    ]


def _layout_kind(layout_type: lldb.SBType) -> str | None:
    name = layout_type.GetDisplayTypeName() or layout_type.GetName() or ""
    for kind in _LAYOUT_KINDS:
        if name == f"cuda::std::{kind}":
            return kind
    return None


def _strides(mapping_ebco: lldb.SBValue, rank: int) -> list[int] | None:
    stride_value = _ebco_element(mapping_ebco, 1, "__strides")
    if stride_value is None:
        return None
    vals = stride_value.GetChildMemberWithName("__vals_")
    if not vals.IsValid():
        return None
    return [vals.GetChildAtIndex(i).GetValueAsSigned(0) for i in range(rank)]


def _required_span_size(
    kind: str | None, extents: list[int], strides: list[int] | None
) -> int:
    """Return the element count spanned by the mapping, mirroring
    layout::mapping::required_span_size(). This can exceed the element count
    (product of extents) for a padded layout_stride mapping."""
    if not extents:
        return 1
    if any(extent == 0 for extent in extents):
        return 0
    if kind == "layout_stride":
        if strides is None:
            return 0
        return 1 + sum(
            (extent - 1) * stride for extent, stride in zip(extents, strides)
        )
    size = 1
    for extent in extents:
        size *= extent
    return size


def _stage_host_copy(
    frame: lldb.SBFrame, data_address: int, byte_count: int
) -> lldb.SBValue | None:
    """Copy byte_count bytes from data_address into inferior-heap memory
    (mirrors cuda::buffer's printer), since it may be device memory LLDB
    would silently read as zeros. Returns a void* SBValue, or None on failure.
    """
    if not frame.IsValid():
        return None
    options = lldb.SBExpressionOptions()
    options.SetIgnoreBreakpoints(True)
    options.SetUnwindOnError(True)
    host_copy = frame.EvaluateExpression(f"(void*)malloc({byte_count})", options)
    if not host_copy.IsValid() or host_copy.GetError().Fail():
        return None
    host_address = host_copy.GetValueAsUnsigned(0)
    if not host_address:
        return None
    result = frame.EvaluateExpression(
        "(int)cudaMemcpy((void*)"
        f"{host_address:#x}, (const void*){data_address:#x}, {byte_count}, "
        f"{_CUDA_MEMCPY_DEFAULT})",
        options,
    )
    if (
        not result.IsValid()
        or result.GetError().Fail()
        or result.GetValueAsSigned(-1) != 0
    ):
        _release_host_copy(frame, host_address)
        return None
    return host_copy


def _release_host_copy(frame: lldb.SBFrame, address: int) -> None:
    if not frame.IsValid() or not address:
        return
    options = lldb.SBExpressionOptions()
    options.SetIgnoreBreakpoints(True)
    options.SetUnwindOnError(True)
    frame.EvaluateExpression(f"(void)free((void*){address:#x})", options)


def _offset(
    kind: str, extents: list[int], strides: list[int] | None, indices: tuple[int, ...]
) -> int:
    if kind == "layout_stride":
        # Rank 0 has no indices to stride over, so strides may legitimately be None.
        if not indices:
            return 0
        return sum(index * stride for index, stride in zip(indices, strides))
    positions = (
        range(len(extents)) if kind == "layout_right" else reversed(range(len(extents)))
    )
    result = 0
    for pos in positions:
        result = result * extents[pos] + indices[pos]
    return result


class MdspanInfo(NamedTuple):
    type_name: str
    extents: list[int] | None
    data: lldb.SBValue | None
    layout: str | None
    strides: list[int] | None

    def size(self) -> int:
        size = 1
        for extent in self.extents:
            size *= extent
        return size


def _readable_type_name(name: str, dynamic_extent: int) -> str:
    """Replace the raw dynamic_extent sentinel with a readable name."""
    return re.sub(rf"\b{dynamic_extent}\b", "dynamic_extent", name)


def _mdspan_info(value: lldb.SBValue) -> MdspanInfo | None:
    value = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
    type_name = cccl_common.canonical_type_name(value.GetType())

    top_ebco = _find_ebco_base(value)
    if top_ebco is None:
        return None
    data_handle = _ebco_element(top_ebco, 0, "__data_handle")
    mapping = _ebco_element(top_ebco, 1, "__mapping")
    if data_handle is None or mapping is None:
        return MdspanInfo(type_name, None, None, None, None)

    mdspan_type = _mdspan_base_type(
        value.GetType().GetCanonicalType().GetUnqualifiedType()
    )
    if mdspan_type is None:
        return MdspanInfo(type_name, None, None, None, None)
    extents_type = mdspan_type.GetTemplateArgumentType(1)
    layout_type = mdspan_type.GetTemplateArgumentType(2)

    dynamic_extent = _dynamic_extent(value.GetTarget())
    static_values = _static_extents(extents_type)
    if static_values is None:
        return MdspanInfo(type_name, None, None, None, None)
    rank_dynamic = sum(1 for value_ in static_values if value_ == dynamic_extent)

    mapping_ebco = _find_ebco_base(mapping)
    dynamic_values: list[int] = []
    if rank_dynamic > 0:
        if mapping_ebco is None:
            return MdspanInfo(type_name, None, None, None, None)
        extents_value = _ebco_element(mapping_ebco, 0, "__extents")
        if extents_value is None:
            return MdspanInfo(type_name, None, None, None, None)
        values = _dynamic_values(extents_value, rank_dynamic)
        if values is None:
            return MdspanInfo(type_name, None, None, None, None)
        dynamic_values = values
    extents = _combined_extents(static_values, dynamic_values, dynamic_extent)

    layout = _layout_kind(layout_type)
    strides = None
    if layout == "layout_stride" and len(extents) > 0:
        if mapping_ebco is None:
            return MdspanInfo(type_name, extents, None, layout, None)
        strides = _strides(mapping_ebco, len(extents))

    data = None
    if data_handle.GetType().GetCanonicalType().IsPointerType():
        data = data_handle

    return MdspanInfo(type_name, extents, data, layout, strides)


def mdspan_summary(value: lldb.SBValue, _internal_dict: InternalDict) -> str | None:
    # Report metadata only (mirrors buffer_summary). Whether the data is
    # readable is already visible in the element list the synthetic provider
    # supplies, so do not stage a throwaway host copy to find out: that costs
    # three inferior calls per print and scales with the element count.
    info = _mdspan_info(cccl_common.strip_reference_value(value))
    if info is None:
        return None
    details = []
    if info.extents is not None:
        extents_text = ", ".join(str(extent) for extent in info.extents)
        details.append(f"rank={len(info.extents)}")
        details.append(f"extents=[{extents_text}]")
    if info.data is not None:
        details.append(f"data={info.data.GetValueAsUnsigned(0):#x}")
    if not details:
        return None
    return ", ".join(details)


class MdspanSyntheticProvider:
    """Expose cuda::std::mdspan elements as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.info: MdspanInfo | None = None
        self.host_copy: lldb.SBValue = lldb.SBValue()
        self.stop_id: int | None = None
        self.update()

    def __del__(self) -> None:
        self._clear_copy()

    def _clear_copy(self) -> None:
        if self.host_copy.IsValid():
            _release_host_copy(
                self.value.GetFrame(), self.host_copy.GetValueAsUnsigned(0)
            )
        self.host_copy = lldb.SBValue()

    def _current_stop_id(self) -> int | None:
        process = self.value.GetProcess()
        return process.GetStopID() if process.IsValid() else None

    def _copy_to_host(self) -> None:
        """Stage the mapping's addressed elements into debugger-owned host
        memory via an inferior cudaMemcpy call. See _stage_host_copy for the
        rationale (real device memory reads as zeros, not an error)."""
        if (
            self.info is None
            or self.info.data is None
            or self.info.extents is None
            or self.info.layout is None
        ):
            return
        span = _required_span_size(
            self.info.layout, self.info.extents, self.info.strides
        )
        if span == 0:
            return
        element_type = self.info.data.GetType().GetPointeeType()
        byte_count = span * element_type.GetByteSize()
        host_copy = _stage_host_copy(
            self.value.GetFrame(), self.info.data.GetValueAsUnsigned(0), byte_count
        )
        if host_copy is None:
            return
        self.host_copy = host_copy.Cast(element_type.GetPointerType())

    def update(self) -> bool:
        # Recopy only when the process has stopped again since the last
        # copy: update() runs before every read (mirrors BufferSyntheticProvider).
        if self.stop_id is not None and self._current_stop_id() == self.stop_id:
            return True
        self._clear_copy()
        self.info = _mdspan_info(self.value)
        self._copy_to_host()
        self.stop_id = self._current_stop_id()
        return True

    def _element_at(self, indices: tuple[int, ...]) -> lldb.SBValue:
        element_type = self.host_copy.GetType().GetPointeeType()
        offset = _offset(
            self.info.layout, self.info.extents, self.info.strides, indices
        )
        byte_offset = offset * element_type.GetByteSize()
        return self.host_copy.CreateChildAtOffset(
            self._label(indices), byte_offset, element_type
        )

    @staticmethod
    def _label(indices: tuple[int, ...]) -> str:
        return "[" + ",".join(str(index) for index in indices) + "]"

    def num_children(self) -> int:
        if self.info is None or not self.host_copy.IsValid():
            return 0
        return self.info.size()

    def has_children(self) -> bool:
        return self.num_children() != 0

    def get_type_name(self) -> str:
        name = cccl_common.canonical_type_name(self.value.GetType())
        dynamic_extent = _dynamic_extent(self.value.GetTarget())
        return _readable_type_name(name, dynamic_extent)

    def get_child_index(self, name: str) -> int:
        if self.info is None or not self.host_copy.IsValid():
            return -1
        stripped = name.strip("[]")
        try:
            parsed = (
                tuple(int(index) for index in stripped.split(",")) if stripped else ()
            )
        except ValueError:
            return -1
        if len(parsed) != len(self.info.extents):
            return -1
        if any(
            index < 0 or index >= extent
            for extent, index in zip(self.info.extents, parsed)
        ):
            return -1
        flat = 0
        for extent, index in zip(self.info.extents, parsed):
            flat = flat * extent + index
        return flat

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if self.info is None or not self.host_copy.IsValid():
            return None
        if index < 0 or index >= self.info.size():
            return None
        extents = self.info.extents
        indices = [0] * len(extents)
        remaining = index
        for pos in reversed(range(len(extents))):
            indices[pos] = remaining % extents[pos]
            remaining //= extents[pos]
        return self._element_at(tuple(indices))


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register the cuda::std::mdspan formatter in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function {module}.mdspan_summary "
        f"--recognizer-function {module}.is_cuda_mdspan"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.MdspanSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_mdspan"
    )
