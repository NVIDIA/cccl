# Copyright (c) 2025CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    from cuda.cooperative.experimental._common import dim3

# Type alias for dimension parameters that can be passed to CUDA functions
DimType = Union["dim3", int, Tuple[int, int], Tuple[int, int, int]]
