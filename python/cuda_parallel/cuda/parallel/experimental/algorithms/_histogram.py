# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import cache_with_key
from .._cccl_interop import call_build, set_cccl_iterator_state, to_cccl_value_state
from .._utils.protocols import get_data_pointer, get_dtype, validate_and_get_stream
from ..iterators._iterators import IteratorBase
from ..typing import DeviceArrayLike


def make_cache_key(
    d_samples: DeviceArrayLike | IteratorBase,
    d_histogram: DeviceArrayLike,
    d_num_output_levels: DeviceArrayLike,
    h_levels: np.ndarray,
    num_samples: int,
):
    d_samples_key = (
        d_samples.kind if isinstance(d_samples, IteratorBase) else get_dtype(d_samples)
    )

    d_histogram_key = get_dtype(d_histogram)
    d_num_output_levels_key = get_dtype(d_num_output_levels)
    d_levels_key = h_levels.dtype

    return (
        d_samples_key,
        d_histogram_key,
        d_num_output_levels_key,
        d_levels_key,
        num_samples,
    )


class _Histogram:
    __slots__ = [
        "num_rows",
        "d_samples_cccl",
        "d_histogram_cccl",
        "h_num_output_levels_cccl",
        "h_lower_level_cccl",
        "h_upper_level_cccl",
        "build_result",
    ]

    def __init__(
        self,
        d_samples: DeviceArrayLike | IteratorBase,
        d_histogram: DeviceArrayLike,
        h_num_output_levels: np.ndarray,
        h_levels: np.ndarray,
        num_samples: int,
    ):
        num_channels = 1
        num_active_channels = 1
        is_evenly_segmented = True
        self.num_rows = 1
        num_levels = h_num_output_levels[0]
        row_stride_samples = num_samples

        self.d_samples_cccl = cccl.to_cccl_iter(d_samples)
        self.d_histogram_cccl = cccl.to_cccl_iter(d_histogram)
        self.h_num_output_levels_cccl = cccl.to_cccl_value(h_num_output_levels)
        self.h_lower_level_cccl = cccl.to_cccl_value(h_levels)
        self.h_upper_level_cccl = cccl.to_cccl_value(h_levels)

        self.build_result = call_build(
            _bindings.DeviceHistogramBuildResult,
            num_channels,
            num_active_channels,
            self.d_samples_cccl,
            num_levels,
            self.d_histogram_cccl,
            self.h_lower_level_cccl,
            self.num_rows,
            row_stride_samples,
            is_evenly_segmented,
        )

    def __call__(
        self,
        temp_storage,
        d_samples: DeviceArrayLike | IteratorBase,
        d_histogram: DeviceArrayLike,
        h_num_output_levels: np.ndarray,
        h_lower_level: np.ndarray,
        h_upper_level: np.ndarray,
        num_samples: int,
        stream=None,
    ):
        set_cccl_iterator_state(self.d_samples_cccl, d_samples)
        set_cccl_iterator_state(self.d_histogram_cccl, d_histogram)
        self.h_num_output_levels_cccl.state = to_cccl_value_state(h_num_output_levels)
        self.h_lower_level_cccl.state = to_cccl_value_state(h_lower_level)
        self.h_upper_level_cccl.state = to_cccl_value_state(h_upper_level)

        stream_handle = validate_and_get_stream(stream)
        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            # Note: this is slightly slower, but supports all ndarray-like objects as long as they support CAI
            # TODO: switch to use gpumemoryview once it's ready
            d_temp_storage = get_data_pointer(temp_storage)

        temp_storage_bytes = self.build_result.compute_even(
            d_temp_storage,
            temp_storage_bytes,
            self.d_samples_cccl,
            self.d_histogram_cccl,
            self.h_num_output_levels_cccl,
            self.h_lower_level_cccl,
            self.h_upper_level_cccl,
            num_samples,
            self.num_rows,
            num_samples,
            stream_handle,
        )

        return temp_storage_bytes


@cache_with_key(make_cache_key)
def histogram(
    d_samples: DeviceArrayLike | IteratorBase,
    d_histogram: DeviceArrayLike,
    h_num_output_levels: np.ndarray,
    h_levels: np.ndarray,
    num_samples: int,
):
    return _Histogram(
        d_samples, d_histogram, h_num_output_levels, h_levels, num_samples
    )
