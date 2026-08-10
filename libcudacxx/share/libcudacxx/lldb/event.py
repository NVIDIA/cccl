# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""LLDB pretty printers for cuda event types."""

from __future__ import annotations

import re

import cccl_common

import lldb

_EVENT_PATTERN = re.compile(r"^cuda::(?:event|event_ref|timed_event)$")
_CHILD_NAME = "handle"
InternalDict = dict[str, object]


def _event_type_name(value_type: lldb.SBType) -> str | None:
    type_name = cccl_common.canonical_type_name(value_type)
    if _EVENT_PATTERN.fullmatch(type_name) is not None:
        return type_name
    return None


def is_cuda_event(value_type: lldb.SBType, _internal_dict: InternalDict) -> bool:
    return _event_type_name(value_type) is not None


def _event_handle(value: lldb.SBValue) -> lldb.SBValue:
    return (
        cccl_common.strip_reference_value(value)
        .GetNonSyntheticValue()
        .GetChildMemberWithName("__event_")
    )


class EventSyntheticProvider:
    """Expose the native handle stored by a cuda event type."""

    def __init__(self, value: lldb.SBValue, _internal_dict: InternalDict) -> None:
        value = cccl_common.strip_reference_value(value)
        self.value = value.GetNonSyntheticValue()
        self.handle = lldb.SBValue()
        self.type_name = ""
        self.update()

    def update(self) -> bool:
        self.type_name = _event_type_name(self.value.GetType()) or ""
        self.handle = _event_handle(self.value)
        return self.handle.IsValid()

    def num_children(self) -> int:
        return int(self.handle.IsValid())

    def has_children(self) -> bool:
        return self.handle.IsValid()

    def get_type_name(self) -> str:
        return self.type_name

    def get_child_index(self, name: str) -> int:
        return 0 if name == _CHILD_NAME else -1

    def get_child_at_index(self, index: int) -> lldb.SBValue | None:
        if index == 0 and self.handle.IsValid():
            return self.handle.Clone(_CHILD_NAME)
        return None


def register(debugger: lldb.SBDebugger, category: str, module: str) -> None:
    """Register cuda event formatters in an LLDB category."""
    debugger.HandleCommand(
        f"type synthetic add --category {category} --python-class {module}.EventSyntheticProvider "
        f"--recognizer-function {module}.is_cuda_event"
    )
