# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Round-trip tests for histogram_even serialize / deserialize."""

import cupy as cp
import numpy as np

from cuda.compute import make_histogram_even
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer
from cuda.compute.algorithms._histogram import _Histogram

try:
    from cuda.compute._build_info import USING_V2
except ImportError:
    USING_V2 = False

import pytest

pytestmark = pytest.mark.skipif(
    USING_V2, reason="AoT not supported on v2 (HostJIT) backend"
)


def _run(
    histogram,
    *,
    d_samples,
    d_histogram,
    h_num_output_levels,
    h_lower_level,
    h_upper_level,
    num_samples,
):
    bytes_needed = histogram(
        temp_storage=None,
        d_samples=d_samples,
        d_histogram=d_histogram,
        h_num_output_levels=h_num_output_levels,
        h_lower_level=h_lower_level,
        h_upper_level=h_upper_level,
        num_samples=num_samples,
    )
    tmp = TempStorageBuffer(bytes_needed, None)
    histogram(
        temp_storage=tmp,
        d_samples=d_samples,
        d_histogram=d_histogram,
        h_num_output_levels=h_num_output_levels,
        h_lower_level=h_lower_level,
        h_upper_level=h_upper_level,
        num_samples=num_samples,
    )


def test_serialize_deserialize_histogram_even_round_trip():
    num_samples = 10
    h_samples = np.array(
        [2.2, 6.1, 7.1, 2.9, 3.5, 0.3, 2.9, 2.1, 6.1, 999.5], dtype="float32"
    )
    d_samples = cp.asarray(h_samples)
    num_levels = 7
    d_histogram = cp.empty(num_levels - 1, dtype="int32")
    h_num_output_levels = np.array([num_levels], dtype=np.int32)
    h_lower_level = np.array([0.0], dtype=np.float32)
    h_upper_level = np.array([12.0], dtype=np.float32)

    builder = make_histogram_even(
        d_samples=d_samples,
        d_histogram=d_histogram,
        h_num_output_levels=h_num_output_levels,
        h_lower_level=h_lower_level,
        h_upper_level=h_upper_level,
        num_samples=num_samples,
    )
    blob = builder.serialize()
    assert len(blob) > 0

    loaded = _Histogram.deserialize(
        blob,
        d_samples=d_samples,
        d_histogram=d_histogram,
        h_num_output_levels=h_num_output_levels,
        h_lower_level=h_lower_level,
        h_upper_level=h_upper_level,
    )
    _run(
        loaded,
        d_samples=d_samples,
        d_histogram=d_histogram,
        h_num_output_levels=h_num_output_levels,
        h_lower_level=h_lower_level,
        h_upper_level=h_upper_level,
        num_samples=num_samples,
    )

    expected, _ = np.histogram(
        h_samples,
        bins=num_levels - 1,
        range=(float(h_lower_level[0]), float(h_upper_level[0])),
    )
    np.testing.assert_array_equal(d_histogram.get(), expected.astype(np.int32))
