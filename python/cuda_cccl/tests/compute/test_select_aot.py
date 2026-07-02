# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Round-trip test for select serialize/deserialize.

select is a composite: it wraps a three_way_partition, so its AoT blob nests
the partitioner's blob. deserialize takes only the blob — no objects.
"""

import cupy as cp
import numpy as np
import pytest

from cuda.compute import deserialize, make_select, serialize
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

pytestmark = pytest.mark.skipif(
    USING_V2, reason="AoT not supported on v2 (HostJIT) backend"
)


def _even(x):
    return x % 2 == 0


def test_serialize_deserialize_select_round_trip():
    n = 1024
    h_in = np.arange(n, dtype=np.int32)
    d_in = cp.asarray(h_in)
    d_out = cp.empty_like(d_in)
    d_num_selected = cp.empty(2, dtype=np.uint64)

    builder = make_select(
        d_in=d_in, d_out=d_out, d_num_selected_out=d_num_selected, cond=_even
    )
    blob = serialize(builder)
    assert len(blob) > 0

    loaded = deserialize(blob)

    def _run():
        nbytes = loaded(
            temp_storage=None,
            d_in=d_in,
            d_out=d_out,
            d_num_selected_out=d_num_selected,
            cond=_even,
            num_items=n,
        )
        loaded(
            temp_storage=TempStorageBuffer(nbytes, None),
            d_in=d_in,
            d_out=d_out,
            d_num_selected_out=d_num_selected,
            cond=_even,
            num_items=n,
        )

    _run()

    k = int(d_num_selected[0].get())
    expected = h_in[h_in % 2 == 0]
    assert k == expected.size
    np.testing.assert_array_equal(d_out.get()[:k], expected)
