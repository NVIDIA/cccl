# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Portable cooperative primitives shared by supported CUDA Python DSLs."""

from ._core._errors import (
    CoopCompilerContextRequiredError as CoopCompilerContextRequiredError,
)
from ._core.api.reduce import reduce as reduce
from ._core.api.reduce import sum as sum
from ._core.api.thread_group import (
    ThreadGroup as ThreadGroup,
)
from ._core.api.thread_group import (
    this_block as this_block,
)
from ._core.api.thread_group import (
    this_warp as this_warp,
)

__version__: str

__all__ = [
    "__version__",
    "CoopCompilerContextRequiredError",
    "ThreadGroup",
    "this_block",
    "this_warp",
    "reduce",
    "sum",
]
