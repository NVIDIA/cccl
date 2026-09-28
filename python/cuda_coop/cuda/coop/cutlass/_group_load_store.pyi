# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Share common Load/Store type hints with the CUTLASS namespace.

Tracing checks group and algorithm support, memory layout, and dtype.
"""

from .._core.api.load_store import load as load
from .._core.api.load_store import store as store
