# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum
from typing import Tuple

from ...typing import DeviceArrayLike


class SortOrder(Enum):
    ASCENDING = 0
    DESCENDING = 1


class DoubleBuffer:
    def __init__(self, d_current: DeviceArrayLike, d_alternate: DeviceArrayLike):
        self.d_buffers = [d_current, d_alternate]
        self.selector = 0

    def current(self):
        return self.d_buffers[self.selector]

    def alternate(self):
        return self.d_buffers[1 - self.selector]


def _get_arrays(
    d_in_keys: DeviceArrayLike | DoubleBuffer,
    d_out_keys: DeviceArrayLike | None,
    d_in_values: DeviceArrayLike | DoubleBuffer | None,
    d_out_values: DeviceArrayLike | None,
) -> Tuple[DeviceArrayLike, DeviceArrayLike, DeviceArrayLike, DeviceArrayLike]:
    if isinstance(d_in_keys, DoubleBuffer):
        d_in_keys_array = d_in_keys.current()
        d_out_keys_array = d_in_keys.alternate()

        if d_in_values is not None:
            assert isinstance(d_in_values, DoubleBuffer)
            d_in_values_array = d_in_values.current()
            d_out_values_array = d_in_values.alternate()
        else:
            d_in_values_array = None
            d_out_values_array = None
    else:
        d_in_keys_array = d_in_keys
        d_in_values_array = d_in_values
        d_out_keys_array = d_out_keys
        d_out_values_array = d_out_values

    return d_in_keys_array, d_out_keys_array, d_in_values_array, d_out_values_array
