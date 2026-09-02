# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import-light portable API organized by cooperative primitive family.

Leaf modules own public argument capture and validation; this facade preserves
the documented root export order and compiler-backend marker contract. It does
not own semantic lowering, provider rendering, or backend compiler state.
"""

from .._errors import CoopCompilerContextRequiredError
from .reduce import reduce, sum
from .thread_group import ThreadGroup, this_block, this_warp

for _member_name in ("this_block", "this_warp", "reduce", "sum"):
    globals()[_member_name].__cuda_coop_backend_member__ = _member_name
del _member_name

__all__ = [
    "CoopCompilerContextRequiredError",
    "ThreadGroup",
    "this_block",
    "this_warp",
    "reduce",
    "sum",
]
