# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Import-light portable API organized by cooperative primitive family.

Leaf modules own public argument capture and validation; this facade preserves
the documented root export order and compiler-backend marker contract. It does
not own semantic lowering, provider rendering, or backend compiler state.
"""

from ._dispatch import (
    _ADJACENT_DIFFERENCE_DIRECTIONS as _ADJACENT_DIFFERENCE_DIRECTIONS,
)
from ._dispatch import (
    _DISCONTINUITY_MODES as _DISCONTINUITY_MODES,
)
from ._dispatch import (
    _EXCHANGE_MODES as _EXCHANGE_MODES,
)
from ._dispatch import (
    _GROUP_OPERATIONS,
)
from ._dispatch import (
    _HISTOGRAM_ALGORITHMS as _HISTOGRAM_ALGORITHMS,
)
from ._dispatch import (
    _LOAD_STORE_ALGORITHMS as _LOAD_STORE_ALGORITHMS,
)
from ._dispatch import (
    _REDUCE_ALGORITHMS as _REDUCE_ALGORITHMS,
)
from ._dispatch import (
    _SCAN_ALGORITHMS as _SCAN_ALGORITHMS,
)
from ._dispatch import (
    _SCAN_MODES as _SCAN_MODES,
)
from ._dispatch import (
    _SHUFFLE_MODES as _SHUFFLE_MODES,
)
from ._dispatch import (
    _backend_module_name as _backend_module_name,
)
from ._dispatch import (
    _common_root_operation_name as _common_root_operation_name,
)
from ._dispatch import (
    _compiler_scope as _compiler_scope,
)
from ._dispatch import (
    _register_qualified_backend as _register_qualified_backend,
)
from ._dispatch import (
    _validate_portable_operation_group as _validate_portable_operation_group,
)
from .adjacent_difference import adjacent_difference as adjacent_difference
from .discontinuity import discontinuity as discontinuity
from .exchange import exchange as exchange
from .histogram import histogram as histogram
from .load_store import load as load
from .load_store import store as store
from .merge_sort import merge_sort_keys as merge_sort_keys
from .merge_sort import merge_sort_pairs as merge_sort_pairs
from .radix import radix_rank as radix_rank
from .radix import radix_sort_keys as radix_sort_keys
from .radix import radix_sort_pairs as radix_sort_pairs
from .reduce import reduce as reduce
from .reduce import sum as sum
from .run_length_decode import run_length_decode as run_length_decode
from .scan import (
    exclusive_scan as exclusive_scan,
)
from .scan import (
    exclusive_sum as exclusive_sum,
)
from .scan import (
    inclusive_scan as inclusive_scan,
)
from .scan import (
    inclusive_sum as inclusive_sum,
)
from .scan import (
    scan as scan,
)
from .shuffle import shuffle as shuffle
from .temp_storage import TempStorage, TempStorageLike
from .thread_data import ThreadData, ThreadDataLike
from .thread_group import (
    Hierarchy,
    ThreadGroup,
    ThreadHierarchy,
    this_block,
    this_cluster,
    this_grid,
    this_thread,
    this_warp,
)
from .topk import (
    topk_max_keys as topk_max_keys,
)
from .topk import (
    topk_max_pairs as topk_max_pairs,
)
from .topk import (
    topk_min_keys as topk_min_keys,
)
from .topk import (
    topk_min_pairs as topk_min_pairs,
)

for _member_name in (
    "TempStorage",
    "ThreadData",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    *_GROUP_OPERATIONS,
):
    setattr(
        globals()[_member_name],
        "__cuda_coop_backend_member__",
        _member_name,
    )
del _member_name

__all__ = [
    "Hierarchy",
    "TempStorage",
    "TempStorageLike",
    "ThreadData",
    "ThreadDataLike",
    "ThreadGroup",
    "ThreadHierarchy",
    "this_block",
    "this_cluster",
    "this_grid",
    "this_thread",
    "this_warp",
    *_GROUP_OPERATIONS,
]
