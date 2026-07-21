# Copyright (c) 2025CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import TYPE_CHECKING, Callable, Literal, Tuple, Union

if TYPE_CHECKING:
    import numba
    import numpy as np

    from ._common import dim3

# Type alias for dimension parameters that can be passed to CUDA functions.
DimType = Union["dim3", int, Tuple[int, int], Tuple[int, int, int]]
"""
.. _DimType:

Dimension parameter specification for CUDA functions.

This type alias accepts the following forms:

- A single integer (1D thread/block configuration).
- A tuple of two or three integers representing multi-dimensional
  (2D/3D) CUDA dimensions.
- A ``dim3`` object, which is a namedtuple with three fields: ``x``,
  ``y``, and ``z``.

Examples
--------
.. code-block:: python

    threads = 128  # 1D
    threads = (16, 16)  # 2D
    threads = (8, 8, 4)  # 3D

See Also
--------
- `CUDA dim3 documentation <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3>`_
"""

DtypeType = Union[str, type, "np.number", "np.dtype", "numba.types.Type"]
"""
.. _DtypeType:

Data type specification for CUDA kernel parameters and operations.

Acceptable forms include:

- Built-in Python numeric types (e.g., ``int``, ``float``).
- NumPy scalar types (e.g., ``np.float32``) or NumPy dtypes (e.g.,
  ``np.dtype('float32')``).
- Numba types (e.g., ``numba.int32``).
- Strings describing the data type (e.g., ``"float32"`` or ``"int64"``).

Examples
--------
.. code-block:: python

    dtype = np.float32
    dtype = "int32"
    dtype = numba.types.float64

See Also
--------
- :mod:`numpy`
- :mod:`numba.types`
"""

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
    Callable[["np.number", "np.number"], "np.number"],
]
"""
.. _ScanOpType:

Specification for binary scan operator used in block-wide CUDA scan operations.

This type can be one of the following:

- **Named operators** (as strings): ``"add"``, ``"mul"``, ``"min"``,
  ``"max"``, ``"bit_and"``, ``"bit_or"``, and ``"bit_xor"``.
- **Symbolic short-hand operators**: ``"+"``, ``"*"``, ``"&"``,
  ``"|"``, and ``"^"``.
- **User-defined callable** that accepts two input values and returns a single
  output value.

"""
