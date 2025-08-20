# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Union

import numpy as np

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import cache_with_key
from .._cccl_interop import call_build, set_cccl_iterator_state, to_cccl_value_state
from .._utils.protocols import get_data_pointer, get_dtype, validate_and_get_stream
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..iterators._iterators import IteratorBase
from ..typing import DeviceArrayLike


def make_cache_key(
    d_samples: DeviceArrayLike | IteratorBase,
    d_histogram: DeviceArrayLike,
    d_num_output_levels: DeviceArrayLike,
    h_lower_level: np.ndarray,
    h_upper_level: np.ndarray,
    num_samples: int,
):
    d_samples_key = (
        d_samples.kind if isinstance(d_samples, IteratorBase) else get_dtype(d_samples)
    )

    d_histogram_key = get_dtype(d_histogram)
    d_num_output_levels_key = get_dtype(d_num_output_levels)
    d_lower_level_key = h_lower_level.dtype
    d_upper_level_key = h_upper_level.dtype

    return (
        d_samples_key,
        d_histogram_key,
        d_num_output_levels_key,
        d_lower_level_key,
        d_upper_level_key,
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
        h_lower_level: np.ndarray,
        h_upper_level: np.ndarray,
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
        self.h_lower_level_cccl = cccl.to_cccl_value(h_lower_level)
        self.h_upper_level_cccl = cccl.to_cccl_value(h_upper_level)

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
def make_histogram_even(
    d_samples: DeviceArrayLike | IteratorBase,
    d_histogram: DeviceArrayLike,
    h_num_output_levels: np.ndarray,
    h_lower_level: np.ndarray,
    h_upper_level: np.ndarray,
    num_samples: int,
):
    """Implements a device-wide histogram that places ``d_samples`` into evenly-spaced bins.

    Example:
        Below, ``histogram`` is used to bin a sequence of samples.

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_histogram_api.py
          :language: python
          :dedent:
          :start-after: example-begin histogram-even
          :end-before: example-end histogram-even

    Args:
        d_samples: Device array or iterator containing the input samples to be histogrammed
        d_histogram: Device array to store the histogram
        h_num_output_levels: Host array containing the number of output levels
        h_lower_level: Host array containing the lower level
        h_upper_level: Host array containing the upper level
        num_samples: Number of samples to be histogrammed

    Returns:
        A callable object that can be used to perform the histogram
    """
    return _Histogram(
        d_samples,
        d_histogram,
        h_num_output_levels,
        h_lower_level,
        h_upper_level,
        num_samples,
    )


def histogram_even(
    d_samples: DeviceArrayLike | IteratorBase,
    d_histogram: DeviceArrayLike,
    num_output_levels: int,
    lower_level: Union[np.floating, np.integer],
    upper_level: Union[np.floating, np.integer],
    num_samples: int,
    stream=None,
):
    """
    Performs device-wide histogram computation with evenly-spaced bins.

    This function automatically handles temporary storage allocation and execution.

    Args:
        d_samples: Device array or iterator containing the input sequence of data samples
        d_histogram: Device array to store the computed histogram
        num_output_levels: Number of histogram bin levels (num_bins = num_output_levels - 1)
        lower_level: Lower sample value bound (inclusive)
        upper_level: Upper sample value bound (exclusive)
        num_samples: Number of input samples
        stream: CUDA stream for the operation (optional)
    """
    # Histogram can accept multiple channels, with one value per channel for
    # each of these parameters. The API only supports one channel for now but we
    # pass arrays to make_histogram_even to support multiple channels in the
    # future.
    h_num_output_levels = np.array([num_output_levels], dtype=np.int32)
    h_lower_level = np.array([lower_level], dtype=type(lower_level))
    h_upper_level = np.array([upper_level], dtype=type(upper_level))

    histogram = make_histogram_even(
        d_samples,
        d_histogram,
        h_num_output_levels,
        h_lower_level,
        h_upper_level,
        num_samples,
    )
    temp_storage_bytes = histogram(
        None,
        d_samples,
        d_histogram,
        h_num_output_levels,
        h_lower_level,
        h_upper_level,
        num_samples,
        stream,
    )
    temp_storage = TempStorageBuffer(temp_storage_bytes, stream)
    histogram(
        temp_storage,
        d_samples,
        d_histogram,
        h_num_output_levels,
        h_lower_level,
        h_upper_level,
        num_samples,
        stream,
    )
