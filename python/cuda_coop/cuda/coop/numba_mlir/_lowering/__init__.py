# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Primitive-family lowering factories for Numba-CUDA-MLIR.

This package registers exact factory identities for the compiler; it does not
recognize providers by module or function name.
"""

from ._exchange import exchange as exchange
from ._exchange import exchange_flagged as exchange_flagged
from ._exchange import exchange_ranked as exchange_ranked
from ._exchange import warp_exchange as warp_exchange
from ._exchange import warp_exchange_ranked as warp_exchange_ranked
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
from ._shuffle import shuffle_array as shuffle_array
from ._shuffle import shuffle_scalar as shuffle_scalar

__all__: tuple[str, ...] = ()
