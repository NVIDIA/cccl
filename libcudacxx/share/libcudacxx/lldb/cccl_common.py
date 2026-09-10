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


class _StagingCache:
    """Own the host copies that the formatters read.

    LLDB builds a synthetic provider per stop and keeps every one alive, so a
    provider cannot free its own copy. The cache frees a copy when the program
    runs again, because every provider restages after that.
    """

    def __init__(self) -> None:
        self.reset(None)

    def reset(self, process_id: int | None) -> None:
        # A copy belongs to (device address, byte count): two views can address
        # one allocation with different lengths, and both stay readable.
        self.copies: dict[tuple[int, int], int] = {}
        self.process_id = process_id
        self.installed = False
        self.stop_id: int | None = None

    def call(self, value: lldb.SBValue, expression: str) -> lldb.SBValue:
        result = evaluate(value, expression)
        process = value.GetProcess()
        if process.IsValid():
            self.stop_id = process.GetStopID()
        return result

    def install(
        self, value: lldb.SBValue, frame: lldb.SBFrame, target: lldb.SBTarget
    ) -> bool:
        # An unresolved identifier fails the whole unit. A statically linked
        # cudart that the program never calls has no cudaMemcpy symbol.
        if any(
            not target.FindSymbols(name).GetSize()
            for name in ("cudaMemcpy", "malloc", "free")
        ):
            return False
        options = lldb.SBExpressionOptions()
        options.SetIgnoreBreakpoints(True)
        options.SetUnwindOnError(True)
        options.SetTopLevel(True)
        options.SetLanguage(lldb.eLanguageTypeC_plus_plus)
        frame.EvaluateExpression(_STAGE_DEFINITION, options)
        # A top-level expression has no result, and it goes to the JIT, so it
        # appears in no symbol table. A call is the only proof that it compiled.
        probe = self.call(value, "__cccl_stage((const void*)0, 0ull)")
        if probe.GetError().Fail():
            return False
        # malloc(0) still returns a pointer that nothing else will free.
        host = probe.GetValueAsUnsigned(0)
        if host:
            self.call(value, f"(void)free((void*){host:#x})")
        return True

    def release_if_resumed(self, value: lldb.SBValue, stop_id: int) -> None:
        # Every inferior call advances the stop ID, so call() records the value
        # its own calls produced. A higher one means the user resumed.
        if self.stop_id is None or stop_id == self.stop_id:
            self.stop_id = stop_id
            return
        self.stop_id = stop_id
        for host in list(self.copies.values()):
            self.call(value, f"(void)free((void*){host:#x})")
        self.copies.clear()


_staging = _StagingCache()


def stage_device_memory(value: lldb.SBValue, device: int, bytes_: int) -> int:
    """Copy ``bytes_`` from inferior address ``device`` to the host heap.

    Returns the host address, or 0 when nothing was requested. Reuses the copy
    of an identical request at the same stop.

    Raises
    ------
    StagingError
        If the frame is dead, the staging function does not compile, or the
        inferior allocation or copy fails.
    """
    frame = value.GetFrame()
    process = value.GetProcess()
    target = value.GetTarget()
    if not frame.IsValid() or not process.IsValid() or not target.IsValid():
        raise StagingError("value has no live frame")
    if bytes_ == 0:
        return 0

    if _staging.process_id != process.GetUniqueID():
        # A restarted process reuses heap addresses, so an old entry would free
        # memory it does not own.
        _staging.reset(process.GetUniqueID())
        _staging.installed = _staging.install(value, frame, target)
    if not _staging.installed:
        raise StagingError("staging function is not available")

    _staging.release_if_resumed(value, process.GetStopID())
    host = _staging.copies.get((device, bytes_))
    if host is not None:
        return host

    result = _staging.call(
        value, f"__cccl_stage((const void*){device:#x}, {bytes_}ull)"
    )
    host = result.GetValueAsUnsigned(0)
    if result.GetError().Fail() or not host:
        raise StagingError(f"cannot stage {bytes_} bytes from {device:#x}")
    _staging.copies[(device, bytes_)] = host
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
