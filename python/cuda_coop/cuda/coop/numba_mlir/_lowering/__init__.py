# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Collect private Load/Store and Reduce provider factories.

Importing this package registers each factory and its scratch ABI. Group
planning selects a factory after validating a public call. Calling a factory
specializes device code but does not run the device operation.
"""

from ._load_store import load as load
from ._load_store import store as store
from ._reduce import block_reduce_builtin as block_reduce_builtin
from ._reduce import group_reduce_block as group_reduce_block
from ._reduce import group_reduce_group as group_reduce_group
from ._reduce import group_reduce_none as group_reduce_none
from ._reduce import group_reduce_warp as group_reduce_warp
from ._reduce import reduce as reduce
from ._reduce import sum as sum
from ._reduce import warp_reduce as warp_reduce
from ._reduce import warp_reduce_builtin as warp_reduce_builtin
from ._reduce import warp_sum as warp_sum

__all__: tuple[str, ...] = ()
