# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Type and value helpers shared by the CCCL LLDB pretty printers."""

from __future__ import annotations

import re

import lldb

_ABI_NAMESPACE_PATTERN = re.compile(r"::__(?:\d+|version_bump_ver\d+_)(?=::)")

# Allocate and copy in one compiled function: an inferior round trip costs far
# more than the work. 4 is cudaMemcpyDefault; the expression parser does not
# import the enum constants.
_STAGE_DEFINITION = """
extern "C" void* __cccl_stage(const void* device, unsigned long long bytes)
{
  void* host = ((void* (*)(unsigned long long))malloc)(bytes);
  if (host != 0
      && ((int (*)(void*, const void*, unsigned long long, int))cudaMemcpy)(
           host, device, bytes, 4) != 0) {
    ((void (*)(void*))free)(host);
    host = 0;
  }
  return host;
}
"""


def evaluate(value: lldb.SBValue, expression: str) -> lldb.SBValue:
    """Evaluate an expression in the frame of a value, without side effects."""
    frame = value.GetFrame()
    if not frame.IsValid():
        return lldb.SBValue()
    options = lldb.SBExpressionOptions()
    options.SetIgnoreBreakpoints(True)
    options.SetUnwindOnError(True)
    return frame.EvaluateExpression(expression, options)


class StagingError(RuntimeError):
    """Report a failure to stage inferior memory for a formatter."""


# Keyed by inferior address, not by provider: LLDB builds a synthetic provider
# per stop and keeps every one alive, so a provider can never free its own copy.
# Maps a device address to (host address, size, stop id).
_staged_copies: dict[int, tuple[int, int, int]] = {}
_staged_process_id: int | None = None


def stage_device_memory(value: lldb.SBValue, device: int, bytes_: int) -> int:
    """Copy ``bytes_`` from inferior address ``device`` to the host heap.

    Returns the host address, or 0 when nothing was requested. Reuses the copy
    from an identical request at the same stop, and otherwise frees it first.

    Raises
    ------
    StagingError
        If the frame is dead, the staging function does not compile, or the
        inferior allocation or copy fails.
    """
    global _staged_process_id

    frame = value.GetFrame()
    process = value.GetProcess()
    if not frame.IsValid() or not process.IsValid():
        raise StagingError("value has no live frame")
    if bytes_ == 0:
        return 0

    if _staged_process_id != process.GetUniqueID():
        # A restarted process reuses heap addresses, so a stale entry would free
        # memory it does not own.
        _staged_copies.clear()
        options = lldb.SBExpressionOptions()
        options.SetIgnoreBreakpoints(True)
        options.SetUnwindOnError(True)
        options.SetTopLevel(True)
        options.SetLanguage(lldb.eLanguageTypeC_plus_plus)
        # A top-level expression has no result, so it reports a generic error
        # even when it compiled.
        error = frame.EvaluateExpression(_STAGE_DEFINITION, options).GetError()
        if error.GetType() == lldb.eErrorTypeExpression:
            raise StagingError(f"staging function did not compile: {error}")
        _staged_process_id = process.GetUniqueID()

    # LLDB may build several providers for one value at one stop, and each asks
    # to stage. Repeating the copy would cost a round trip and change nothing.
    stop_id = process.GetStopID()
    previous = _staged_copies.pop(device, None)
    if previous is not None:
        host, staged_bytes, staged_stop = previous
        if (staged_bytes, staged_stop) == (bytes_, stop_id):
            _staged_copies[device] = previous
            return host
        evaluate(value, f"(void)free((void*){host:#x})")

    result = evaluate(value, f"__cccl_stage((const void*){device:#x}, {bytes_}ull)")
    host = result.GetValueAsUnsigned(0)
    if result.GetError().Fail() or not host:
        raise StagingError(f"cannot stage {bytes_} bytes from {device:#x}")
    _staged_copies[device] = (host, bytes_, stop_id)
    return host


def strip_reference(value_type: lldb.SBType) -> lldb.SBType:
    """Return the type behind any reference, typedef, or cv-qualifier.

    GetDereferencedType() is a no-op on a non-reference, so it needs no guard.
    """
    return value_type.GetCanonicalType().GetDereferencedType().GetUnqualifiedType()


def strip_reference_value(value: lldb.SBValue) -> lldb.SBValue:
    """Return the referenced value, or the value itself if it is not a reference."""
    if value.GetType().IsReferenceType():
        return value.Dereference()
    return value


def canonical_type_name(value_type: lldb.SBType) -> str:
    """Return the display name of the type behind any reference or typedef."""
    return strip_reference(value_type).GetDisplayTypeName() or ""


def public_type_name(value_type: lldb.SBType) -> str:
    """Return the complete type name without CUDA ABI inline namespaces.

    GetName() keeps default template arguments that GetDisplayTypeName() hides.
    """
    value_type = strip_reference(value_type)
    type_name = value_type.GetName() or value_type.GetDisplayTypeName() or ""
    return _ABI_NAMESPACE_PATTERN.sub("", type_name)
