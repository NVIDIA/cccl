# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS family lowerers and generated-provider renderers.

Each module translates one public ``_group_<family>`` contract into CUTLASS,
CUB, or CUDAX compiler operations. Public modules import these lowerers lazily
so importing :mod:`cuda.coop.cutlass` does not activate compiler machinery.
"""

__all__: tuple[str, ...] = ()
