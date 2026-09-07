# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS compiler activation and provider lifecycle machinery.

Semantic frontends and family lowerers deliberately live outside this package.
Importing it activates only the lightweight trace-context integration; bundle
compilation, caching, and final linking remain lazy until a primitive is used.
"""

from ._activation import register_trace_context, trace_context

__all__ = ["register_trace_context", "trace_context"]
