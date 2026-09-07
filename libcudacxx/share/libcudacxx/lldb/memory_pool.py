# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printers for cuda memory pool types."""

from __future__ import annotations

import re

import cccl_common

import lldb

_POOL_PATTERN = re.compile(
    r"^cuda::(?:(?:device|managed|pinned)_memory_pool(?:_ref)?"
    r"|shared_(?:device|managed|pinned)_memory_pool)$"
)
_SHARED_POOL_PATTERN = re.compile(
    r"^cuda::shared_(?:device|managed|pinned)_memory_pool$"
)
# Attribute names in the order the snapshot expression returns them.
_POOL_ATTRIBUTE_NAMES = (
    "release_threshold",
    "reserved_mem_current",
    "reserved_mem_high",
    "used_mem_current",
    "used_mem_high",
)
_SNAPSHOT_LANES = len(_POOL_ATTRIBUTE_NAMES) + 1
_ATTRIBUTE_TYPE = "unsigned long long"
# A driver call still running after this long is wedged; print only the handle.
_EXPRESSION_TIMEOUT_US = 1_000_000
_MAX_BASE_DEPTH = 4
InternalDict = dict[str, object]


# LLDB cannot materialize a locally declared struct returned by an expression, so
# a Clang vector keeps this to one inferior call: five attributes plus a validity
# mask. cuMemPoolGetAttribute is the driver entry point CCCL itself uses, and
# unlike the runtime spelling it never creates a primary context.
_POOL_SNAPSHOT_EXPRESSION = """
(unsigned long long __attribute__((ext_vector_type(6))))(([](void* pool) {
  using Snapshot = unsigned long long __attribute__((ext_vector_type(6)));

  unsigned long long release_threshold{};
  unsigned long long reserved_mem_current{};
  unsigned long long reserved_mem_high{};
  unsigned long long used_mem_current{};
  unsigned long long used_mem_high{};
  unsigned long long validity{};

  if (((int (*)(void*, int, void*))cuMemPoolGetAttribute)(
        pool, 4, &release_threshold) == 0) {
    validity |= 1ull << 0;
  }
  if (((int (*)(void*, int, void*))cuMemPoolGetAttribute)(
        pool, 5, &reserved_mem_current) == 0) {
    validity |= 1ull << 1;
  }
  if (((int (*)(void*, int, void*))cuMemPoolGetAttribute)(
        pool, 6, &reserved_mem_high) == 0) {
    validity |= 1ull << 2;
  }
  if (((int (*)(void*, int, void*))cuMemPoolGetAttribute)(
        pool, 7, &used_mem_current) == 0) {
    validity |= 1ull << 3;
  }
  if (((int (*)(void*, int, void*))cuMemPoolGetAttribute)(
        pool, 8, &used_mem_high) == 0) {
    validity |= 1ull << 4;
  }

  return Snapshot{
    release_threshold,
    reserved_mem_current,
    reserved_mem_high,
    used_mem_current,
    used_mem_high,
    validity,
  };
})((void*)%#x))
"""

# The expression hardcodes its lane count, so extending _POOL_ATTRIBUTE_NAMES
# requires rewriting it by hand.
assert _POOL_SNAPSHOT_EXPRESSION.count(f"ext_vector_type({_SNAPSHOT_LANES})") == 2, (
    "_POOL_SNAPSHOT_EXPRESSION lane count must match _POOL_ATTRIBUTE_NAMES"
)


def _pool_type_name(value_type: lldb.SBType) -> str | None:
    type_name = cccl_common.canonical_type_name(value_type)
    if _POOL_PATTERN.fullmatch(type_name) is not None:
        return type_name
    return None


def is_memory_pool(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    return _pool_type_name(value_type) is not None


def _is_shared_pool(value_type: lldb.SBType) -> bool:
    type_name = cccl_common.canonical_type_name(value_type)
    return _SHARED_POOL_PATTERN.fullmatch(type_name) is not None


def _find_member(value: lldb.SBValue, name: str, depth: int = 0) -> lldb.SBValue:
    """Return a member of value or of one of its base classes."""
    value = value.GetNonSyntheticValue()
    member = value.GetChildMemberWithName(name)
    if member.IsValid():
        return member
    if depth >= _MAX_BASE_DEPTH:
        return lldb.SBValue()

    value_type = value.GetType()
    for index in range(value_type.GetNumberOfDirectBaseClasses()):
        base_class = value_type.GetDirectBaseClassAtIndex(index)
        base = value.GetChildMemberWithName(base_class.GetName())
        if not base.IsValid():
            base = value.GetChildAtIndex(index)
        if not base.IsValid():
            continue
        member = _find_member(base, name, depth + 1)
        if member.IsValid():
            return member
    return lldb.SBValue()


def _pool_handle(value: lldb.SBValue) -> lldb.SBValue:
    return _find_member(cccl_common.strip_reference_value(value), "__pool_")


def _nested_member(value: lldb.SBValue, *names: str) -> lldb.SBValue:
    """Walk a chain of direct members, bypassing any synthetic providers."""
    for name in names:
        if not value.IsValid():
            return lldb.SBValue()
        value = value.GetNonSyntheticValue().GetChildMemberWithName(name)
    return value


def _pool_use_count(value: lldb.SBValue) -> int | None:
    """Return the number of owners sharing the pool, or None if unreadable."""
    reference = _find_member(cccl_common.strip_reference_value(value), "__ref_")
    if not reference.IsValid():
        return None
    block = reference.GetNonSyntheticValue().GetChildMemberWithName("__block_")
    if not block.IsValid() or block.GetError().Fail():
        return None
    if block.GetValueAsUnsigned(0) == 0:
        return 0
    # The count is a cuda::std::atomic<int>, stored behind __a/__a_value.
    stored = _nested_member(block.Dereference(), "__ref_count", "__a", "__a_value")
    if not stored.IsValid() or stored.GetError().Fail():
        return None
    return stored.GetValueAsSigned(0)


def _evaluate(value: lldb.SBValue, expression: str) -> lldb.SBValue:
    frame = value.GetFrame()
    if not frame.IsValid():
        return lldb.SBValue()
    options = lldb.SBExpressionOptions()
    options.SetIgnoreBreakpoints(True)
    options.SetUnwindOnError(True)
    # Never resume other threads or hang the debugger on a wedged driver call.
    options.SetTryAllThreads(False)
    options.SetTimeoutInMicroSeconds(_EXPRESSION_TIMEOUT_US)
    return frame.EvaluateExpression(expression, options)


def _query_pool_attributes(
    value: lldb.SBValue, handle: int
) -> tuple[tuple[str, int], ...]:
    result = _evaluate(value, _POOL_SNAPSHOT_EXPRESSION % handle)
    if not result.IsValid() or result.GetError().Fail():
        return ()
    if result.GetNumChildren() != _SNAPSHOT_LANES:
        return ()

    lanes: list[int] = []
    for index in range(_SNAPSHOT_LANES):
        lane = result.GetChildAtIndex(index)
        if not lane.IsValid() or lane.GetError().Fail():
            return ()
        lanes.append(lane.GetValueAsUnsigned(0))

    validity = lanes[-1]
    return tuple(
        (name, lanes[index])
        for index, name in enumerate(_POOL_ATTRIBUTE_NAMES)
        if validity & (1 << index)
    )


def _pool_attributes(value: lldb.SBValue) -> tuple[tuple[str, int], ...]:
    """Return the driver-reported attributes of the pool held by value."""
    handle = _pool_handle(value)
    if not handle.IsValid() or handle.GetError().Fail():
        return ()

    raw_handle = handle.GetValueAsUnsigned(0)
    if raw_handle == 0:
        return ()
    return _query_pool_attributes(value, raw_handle)


def memory_pool_summary(
    value: lldb.SBValue, _internal_dict: InternalDict
) -> str | None:
    value = cccl_common.strip_reference_value(value)
    handle = _pool_handle(value)
    if not handle.IsValid() or handle.GetError().Fail():
        return None

    description = f"handle={handle.GetValueAsUnsigned(0):#x}"
    if _is_shared_pool(value.GetType()):
        use_count = _pool_use_count(value)
        if use_count is not None:
            description += f", use_count={use_count}"
    return description


class MemoryPoolSyntheticProvider:
    """Expose CUDA memory pool attributes as LLDB synthetic children."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        self.value = cccl_common.strip_reference_value(value).GetNonSyntheticValue()
        # Canonicalizing resolves any alias while keeping cv/ref qualifiers.
        self.type_name = value.GetType().GetCanonicalType().GetDisplayTypeName() or ""
        self.children: tuple[tuple[str, int], ...] = ()
        self.stop_id: int | None = None
        self.initialized = False

    def update(self) -> bool:
        # LLDB calls update() before every read, so only query the driver once
        # per stop.
        process = self.value.GetProcess()
        stop_id = process.GetStopID() if process.IsValid() else None
        if self.initialized and stop_id == self.stop_id:
            return True

        self.stop_id = stop_id
        self.initialized = True
        self.children = _pool_attributes(self.value)
        if not self.children:
            return True

        # New children invalidate LLDB's cache; later calls at this stop reuse them.
        return False

    def num_children(self) -> int:
        return len(self.children)

    def has_children(self) -> bool:
        return bool(self.children)

    def get_type_name(self) -> str:
        return self.type_name

    def get_child_index(self, name: str) -> int:
        for index, (child_name, _) in enumerate(self.children):
            if child_name == name:
                return index
        return -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index < 0 or index >= len(self.children):
            return None
        name, attribute_value = self.children[index]
        target = self.value.GetTarget()
        child_type = target.FindFirstType(_ATTRIBUTE_TYPE)
        if not child_type.IsValid():
            return None

        byte_order = target.GetByteOrder()
        python_byte_order = "big" if byte_order == lldb.eByteOrderBig else "little"
        raw_data = attribute_value.to_bytes(
            child_type.GetByteSize(), byteorder=python_byte_order
        )
        data = lldb.SBData()
        error = lldb.SBError()
        data.SetData(error, raw_data, byte_order, target.GetAddressByteSize())
        if error.Fail():
            return None
        return self.value.CreateValueFromData(name, data, child_type)


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register CUDA memory pool formatters in an LLDB category."""
    debugger.HandleCommand(
        f"type summary add --category {category} --expand --python-function "
        f"{module}.memory_pool_summary --recognizer-function {module}.is_memory_pool"
    )
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class "
        f"{module}.MemoryPoolSyntheticProvider --recognizer-function "
        f"{module}.is_memory_pool"
    )
