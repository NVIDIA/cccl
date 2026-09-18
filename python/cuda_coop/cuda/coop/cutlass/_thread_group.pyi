# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing declarations for CUTLASS group descriptors."""

from typing import Generic, Literal, TypeAlias

from typing_extensions import TypeVar

from .. import ThreadHierarchy
from .._core.api.thread_group import ThreadGroup as CommonThreadGroup
from .._typing import ThreadGroupKind

_GroupKindT_co = TypeVar(
    "_GroupKindT_co", bound=ThreadGroupKind, covariant=True, default=ThreadGroupKind
)

Hierarchy = ThreadHierarchy

class ThreadGroup(CommonThreadGroup[_GroupKindT_co], Generic[_GroupKindT_co]):
    """A common group descriptor consumed by CUTLASS primitives."""

BlockGroup: TypeAlias = ThreadGroup[Literal["block"]]
MemoryGroup: TypeAlias = BlockGroup

def this_block() -> BlockGroup:
    """Describe the current CUDA thread block."""

__all__ = ["Hierarchy", "ThreadGroup", "ThreadHierarchy", "this_block"]
