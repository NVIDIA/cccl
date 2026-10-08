# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Declare block Load/Store calls for static type checkers.

Keep annotations broad enough for compiler scalar values. Tracing checks the
block group, algorithm support, memory layout, and dtype.
"""

from typing import Any

from .._core.api import TempStorageLike
from .._core.api.thread_group import ThreadGroup
from ._thread_data import ThreadData

def load(
    group: ThreadGroup,
    source: Any,
    output: ThreadData,
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    oob_default: Any = None,
    offset: Any = None,
    temp_storage: TempStorageLike | None = None,
) -> None: ...
def store(
    group: ThreadGroup,
    destination: Any,
    value: Any,
    /,
    *,
    algorithm: Any = "direct",
    valid_items: Any = None,
    offset: Any = None,
    temp_storage: TempStorageLike | None = None,
) -> None: ...
