# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Round-trip tests for segmented_reduce serialize / deserialize."""

import cupy as cp
import numpy as np

from cuda.compute import OpKind, make_segmented_reduce
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer
from cuda.compute.algorithms._segmented_reduce import _SegmentedReduce


def _run(reducer, *, d_in, d_out, num_segments, start, end, op, h_init):
    bytes_needed = reducer(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        num_segments=num_segments,
        start_offsets_in=start,
        end_offsets_in=end,
        op=op,
        h_init=h_init,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    reducer(
        temp_storage=tmp,
        d_in=d_in,
        d_out=d_out,
        num_segments=num_segments,
        start_offsets_in=start,
        end_offsets_in=end,
        op=op,
        h_init=h_init,
    )


def test_serialize_deserialize_segmented_reduce_round_trip():
    h_in = np.array([8, 6, 7, 5, 3, 0, 9, -4, 3, 0, 1, 3, 1, 11, 25, 8], dtype=np.int32)
    offsets = np.array([0, 7, 11, 16], dtype=np.int64)
    d_in = cp.asarray(h_in)
    start = cp.asarray(offsets[:-1])
    end = cp.asarray(offsets[1:])
    n_segments = start.size
    d_out = cp.empty(n_segments, dtype=cp.int32)
    h_init = np.array([0], dtype=np.int32)

    builder = make_segmented_reduce(
        d_in=d_in,
        d_out=d_out,
        start_offsets_in=start,
        end_offsets_in=end,
        op=OpKind.PLUS,
        h_init=h_init,
    )
    blob = builder.serialize()
    assert len(blob) > 0

    loaded = _SegmentedReduce.deserialize(
        blob,
        d_in=d_in,
        d_out=d_out,
        start_offsets_in=start,
        end_offsets_in=end,
        op=OpKind.PLUS,
        h_init=h_init,
    )
    _run(
        loaded,
        d_in=d_in,
        d_out=d_out,
        num_segments=n_segments,
        start=start,
        end=end,
        op=OpKind.PLUS,
        h_init=h_init,
    )

    expected = np.array(
        [h_in[s:e].sum() for s, e in zip(offsets[:-1], offsets[1:])], dtype=np.int32
    )
    np.testing.assert_array_equal(d_out.get(), expected)
