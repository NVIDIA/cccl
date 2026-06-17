# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import tempfile
from collections import namedtuple
from enum import Enum
from typing import BinaryIO, Union

import numba
import numpy as np

from ._typing import DimType

version = namedtuple("version", ("major", "minor"))
code = namedtuple("code", ("kind", "version", "data"))
symbol = namedtuple("symbol", ("kind", "name"))
dim3 = namedtuple("dim3", ("x", "y", "z"))


CUB_BLOCK_SCAN_ALGOS = {
    "raking": "::cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING",
    "raking_memoize": "::cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE",
    "warp_scans": "::cub::BlockScanAlgorithm::BLOCK_SCAN_WARP_SCANS",
}


CUB_BLOCK_REDUCE_ALGOS = {
    "raking_commutative_only": "::cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY",
    "raking": "::cub::BlockReduceAlgorithm::BLOCK_REDUCE_RAKING",
    "warp_reductions": "::cub::BlockReduceAlgorithm::BLOCK_REDUCE_WARP_REDUCTIONS",
}


class CudaSharedMemConfig(Enum):
    """
    CUDA shared memory configuration.  This is intended to mirror the C++
    equivalent `cudaSharedMemConfig` enum.
    """

    BankSizeDefault = 0
    BankSizeFourByte = 1
    BankSizeEightByte = 2

    def __str__(self):
        return f"cudaSharedMem{self.name}"


def make_binary_tempfile(content: bytes, suffix: str) -> BinaryIO:
    """
    Creates an unbuffered temporary binary file containing **content** and
    ending with **suffix**.  The caller is responsible for closing the file.

    :param content: Supplies the content to write to the temporary file.

    :param suffix: Supplies the suffix for the temporary file.

    :return: A binary file-like object representing the temporary file.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w+b", suffix=suffix, buffering=0, delete=False
    )
    tmp.write(content)
    return tmp


def check_in(name, arg, set):
    if arg not in set:
        raise ValueError(f"{name} must be in {set} ; got {name} = {arg}")


def check_not_in(name, arg, set):
    if arg in set:
        raise ValueError(
            f"{name} must not be any of those value {set} ; got {name} = {arg}"
        )


def check_contains(set, key):
    if key not in set:
        raise ValueError(f"{key} must be in {set}")


def check_dim3(name, arg):
    if len(arg) != 3:
        raise ValueError(f"{name} should be a length-3 tuple ; got {name} = {arg}")


def find_unsigned(name, txt):
    regex = re.compile(f".global .align 4 .u32 {name} = ([0-9]*);", re.MULTILINE)
    found = regex.search(txt)
    if found is None:  # TODO: improve regex logic
        regex = re.compile(f".global .align 4 .u32 {name};", re.MULTILINE)
        found = regex.search(txt)
        if found is not None:
            return 0
        else:
            raise ValueError(f"{name} not found in text")
    else:
        return int(found.group(1))


def find_mangled_name(name, txt):
    regex = re.compile(f"[_a-zA-Z0-9]*{name}[_a-zA-Z0-9]*", re.MULTILINE)
    return regex.search(txt).group(0)


def find_dim2(name, txt):
    return (find_unsigned(f"{name}_x", txt), find_unsigned(f"{name}_y", txt))


def find_dim3(name, txt):
    return (
        find_unsigned(f"{name}_x", txt),
        find_unsigned(f"{name}_y", txt),
        find_unsigned(f"{name}_z", txt),
    )


def normalize_dim_param(dim: DimType) -> dim3:
    """
    Normalize the dim parameter to a `dim3` (x, y, z) instance.

    The logic for this routine is as follows:

    - If the dim is already a `dim3` instance, return it as is.
    - If the dim is a positive integer, return a 1D `dim3` instance with the
      integer value as the x-dimension.  If the dim is a negative integer,
      raise a ValueError.
    - If the dim is a tuple:
        - If the tuple has two elements, return a 2D `dim3` instance with the
          tuple values as the x and y dimensions.  If either value is
          negative, raise a ValueError.
        - If the tuple has three elements, return a 3D `dim3` instance with
          the tuple values as the x, y, and z dimensions.  If any value is
          negative, raise a ValueError.

    Args:
        dim: Supplies the dim parameter to normalize.

    Returns:
        The normalized dim parameter as a `dim3` instance.

    Raises:
        ValueError: If the dim is invalid.

    """
    if isinstance(dim, dim3):
        return dim

    if isinstance(dim, int):
        if dim < 0:
            msg = f"Dimension value must be non-negative, got {dim}"
            raise ValueError(msg)
        return dim3(dim, 1, 1)

    if isinstance(dim, tuple):
        if len(dim) == 2:
            x, y = dim
            z = 1
            if x < 0 or y < 0:
                msg = f"Dimension values must be non-negative, got {dim}"
                raise ValueError(msg)
            return dim3(x, y, z)
        elif len(dim) == 3:
            x, y, z = dim
            if x < 0 or y < 0 or z < 0:
                msg = f"Dimension values must be non-negative, got {dim}"
                raise ValueError(msg)
            return dim3(x, y, z)
        else:
            msg = f"Tuple dimension must have 2 or 3 elements, got {len(dim)}"
            raise ValueError(msg)

    raise ValueError(f"Unsupported dimension type: {type(dim)}")


def normalize_dtype_param(
    dtype: Union[str, type, "np.dtype", "numba.types.Type"],
) -> "numba.types.Type":
    """
    Normalize the dtype parameter to an appropriate Numba type.

    The logic for this routine is as follows:

    - If the dtype is already a numba type, return it as is.
    - If the dtype is a valid numpy dtype, convert it to the corresponding
      numba type.  Note that this applies to both `np.int32` and
      `np.dtype(np.int32)`.
    - If the dtype is a string:
        - If there's a period in the string, ensure it's because the string
          starts with "np." and is followed by a valid numpy dtype.  Otherwise,
          raise a ValueError in both cases: if there's a period belonging to
          something other than the leading "np.", or if the following numpy
          type isn't valid.
        - If there's no period, assume the type is referring to a numba type.
          If not, raise a ValueError.  If it is, return the corresponding numba
          type.

    Args:
        dtype: Supplies the dtype parameter to normalize.

    Returns:
        The normalized dtype parameter as a numba type.

    Raises:
        ValueError: If the dtype is invalid.
    """
    import numba
    import numpy as np

    # If dtype is already a numba type, return it as is.
    if isinstance(dtype, numba.types.Type):
        return dtype

    # Handle numpy dtype objects.
    if hasattr(np, "dtype") and isinstance(dtype, np.dtype):
        # Convert numpy dtype to numba type.
        try:
            return numba.from_dtype(dtype)
        except Exception as e:
            msg = f"Failed to convert numpy dtype {dtype} to numba type: {e}"
            raise ValueError(msg)

    is_type_like = (
        hasattr(np, "generic")
        and isinstance(dtype, type)
        and issubclass(dtype, np.generic)
    )

    # Handle numpy type objects (like np.int32).
    if is_type_like:
        try:
            return numba.from_dtype(np.dtype(dtype))
        except Exception as e:
            msg = f"Failed to convert numpy type {dtype} to numba type: {e}"
            raise ValueError(msg)

    # Handle string representations.
    if isinstance(dtype, str):
        if "." in dtype:
            # Check if string starts with "np." and is followed by a
            # valid numpy dtype.
            if not dtype.startswith("np."):
                msg = (
                    f"Invalid dtype string format: {dtype}. "
                    "String with period must start with 'np.' "
                    "and be followed by a valid numpy dtype."
                )
                raise ValueError(msg)

            # Extract the numpy type name.
            np_type_name = dtype[3:]

            # Check if it's a valid numpy type.
            if not hasattr(np, np_type_name):
                raise ValueError(f"Invalid numpy dtype: {np_type_name}")

            # Convert to numba type.
            try:
                np_type = getattr(np, np_type_name)
                return numba.from_dtype(np.dtype(np_type))
            except Exception as e:
                msg = f"Failed to convert numpy type {np_type_name} to numba type: {e}"
                raise ValueError(msg)
        else:
            # No period, assume it's a numba type.
            if not hasattr(numba, dtype):
                raise ValueError(f"Invalid numba type name: {dtype}")

            return getattr(numba, dtype)

    # If we get here, the dtype is not recognized.
    raise ValueError(f"Unrecognized dtype format: {dtype}")
