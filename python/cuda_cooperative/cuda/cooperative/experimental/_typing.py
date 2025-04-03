# Copyright (c) 2025CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Callable, Literal, Tuple, TypeVar, Union

if TYPE_CHECKING:
    import numba
    import numpy as np

    from cuda.cooperative.experimental._common import dim3

# Type alias for dimension parameters that can be passed to CUDA functions.
DimType = Union["dim3", int, Tuple[int, int], Tuple[int, int, int]]

T = TypeVar("T", bound="np.number")

# Type alias for scan operators.
ScanOpType = Union[
    # Explicitly named operators.
    Literal[
        "add",
        "plus",
        "mul",
        "multiplies",
        "min",
        "minimum",
        "max",
        "maximum",
        "bit_and",
        "bit_or",
        "bit_xor",
    ],
    # Short-hand operators.
    Literal["+", "*", "&", "|", "^"],
    # Callable objects.
    Callable[["numba.types.Number", "numba.types.Number"], "numba.types.Number"],
    Callable[["np.ndarray", "np.ndarray"], "np.ndarray"],
    Callable[[T, T], T],
]
