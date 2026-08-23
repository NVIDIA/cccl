# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import re
import tempfile
from collections import namedtuple
from enum import Enum
from numbers import Integral
from typing import BinaryIO, Union

import numpy as np
from numba_cuda_mlir import types as numba_mlir_types

from cuda.coop._core.dtype_policy import (
    validate_common_v1_integer_key_dtype_name,
    validate_common_v1_integer_value_dtype_name,
    validate_common_v1_numeric_dtype_name,
)

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
    ending with **suffix**.  The returned file is closed; the caller is
    responsible for removing the file when its path is no longer needed.

    :param content: Supplies the content to write to the temporary file.

    :param suffix: Supplies the suffix for the temporary file.

    :return: A binary file-like object representing the temporary file.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w+b", suffix=suffix, buffering=0, delete=False
    )
    try:
        tmp.write(content)
    except Exception:
        name = tmp.name
        tmp.close()
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise
    tmp.close()
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
    escaped_name = re.escape(name)
    regex = re.compile(
        f".global .align 4 .u32 {escaped_name} = ([0-9]*);", re.MULTILINE
    )
    found = regex.search(txt)
    if found is None:  # TODO: improve regex logic
        regex = re.compile(f".global .align 4 .u32 {escaped_name};", re.MULTILINE)
        found = regex.search(txt)
        if found is not None:
            return 0
        else:
            raise ValueError(f"{name} not found in text")
    else:
        return int(found.group(1))


def find_mangled_name(name, txt):
    regex = re.compile(f"[_a-zA-Z0-9]*{re.escape(name)}[_a-zA-Z0-9]*", re.MULTILINE)
    found = regex.search(txt)
    if found is None:
        raise ValueError(f"{name} not found in text")
    return found.group(0)


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
      integer value as the x-dimension.  If the dim is not positive, raise a
      ValueError.
    - If the dim is a tuple:
        - If the tuple has two elements, return a 2D `dim3` instance with the
          tuple values as the x and y dimensions.  If either value is not
          positive, raise a ValueError.
        - If the tuple has three elements, return a 3D `dim3` instance with
          the tuple values as the x, y, and z dimensions.  If any value is not
          positive, raise a ValueError.

    Args:
        dim: Supplies the dim parameter to normalize.

    Returns:
        The normalized dim parameter as a `dim3` instance.

    Raises:
        ValueError: If the dim is invalid.

    """

    def _validate_positive(values):
        if any(value < 1 for value in values):
            raise ValueError(f"Dimension values must be positive, got {dim}")

    if isinstance(dim, dim3):
        _validate_positive(dim)
        return dim

    if isinstance(dim, int):
        _validate_positive((dim,))
        return dim3(dim, 1, 1)

    if isinstance(dim, tuple):
        if len(dim) == 2:
            x, y = dim
            z = 1
            _validate_positive((x, y))
            return dim3(x, y, z)
        elif len(dim) == 3:
            x, y, z = dim
            _validate_positive((x, y, z))
            return dim3(x, y, z)
        else:
            msg = f"Tuple dimension must have 2 or 3 elements, got {len(dim)}"
            raise ValueError(msg)

    raise ValueError(f"Unsupported dimension type: {type(dim)}")


def resolve_threads_per_block_alias(threads_per_block, dim):
    """Resolve the legacy ``dim`` alias used by scoped factory APIs."""
    if threads_per_block is None:
        return dim
    if dim is not None:
        raise ValueError("threads_per_block and dim are aliases; provide only one")
    return threads_per_block


_NP_DTYPE_TO_NUMBA_MLIR_TYPE = {
    np.dtype(np.bool_): numba_mlir_types.boolean,
    np.dtype(np.int8): numba_mlir_types.int8,
    np.dtype(np.int16): numba_mlir_types.int16,
    np.dtype(np.int32): numba_mlir_types.int32,
    np.dtype(np.int64): numba_mlir_types.int64,
    np.dtype(np.uint8): numba_mlir_types.uint8,
    np.dtype(np.uint16): numba_mlir_types.uint16,
    np.dtype(np.uint32): numba_mlir_types.uint32,
    np.dtype(np.uint64): numba_mlir_types.uint64,
    np.dtype(np.float16): numba_mlir_types.float16,
    np.dtype(np.float32): numba_mlir_types.float32,
    np.dtype(np.float64): numba_mlir_types.float64,
    np.dtype(np.complex64): numba_mlir_types.complex64,
    np.dtype(np.complex128): numba_mlir_types.complex128,
}

_NUMBA_MLIR_TYPE_NAME_ALIASES = {
    "bool_": "boolean",
    "bool": "boolean",
}


def _normalize_numba_mlir_type_name(type_name: str) -> str:
    return _NUMBA_MLIR_TYPE_NAME_ALIASES.get(type_name, type_name)


def _dtype_from_numpy(np_dtype: np.dtype) -> numba_mlir_types.Type:
    canonical = np.dtype(np_dtype)
    if canonical in _NP_DTYPE_TO_NUMBA_MLIR_TYPE:
        return _NP_DTYPE_TO_NUMBA_MLIR_TYPE[canonical]

    type_name = _normalize_numba_mlir_type_name(canonical.name)
    if hasattr(numba_mlir_types, type_name):
        resolved = getattr(numba_mlir_types, type_name)
        if isinstance(resolved, numba_mlir_types.Type):
            return resolved

    raise ValueError(f"Unsupported numpy dtype: {canonical}")


def normalize_dtype_param(
    dtype: Union[str, type, "np.dtype", "numba_mlir_types.Type"],
) -> "numba_mlir_types.Type":
    """
    Normalize a dtype parameter into a Numba-CUDA-MLIR type object.
    """
    if dtype is bool:
        return numba_mlir_types.boolean

    if dtype is int:
        return numba_mlir_types.int32

    if dtype is float:
        return numba_mlir_types.float32

    if dtype is complex:
        return numba_mlir_types.complex128

    if isinstance(dtype, numba_mlir_types.Type):
        return dtype

    if isinstance(dtype, np.dtype):
        return _dtype_from_numpy(dtype)

    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        return _dtype_from_numpy(np.dtype(dtype))

    if isinstance(dtype, str):
        if dtype.startswith("np."):
            np_type_name = dtype[3:]
            if not hasattr(np, np_type_name):
                raise ValueError(f"Invalid numpy dtype: {np_type_name}")
            return _dtype_from_numpy(np.dtype(getattr(np, np_type_name)))

        for prefix in ("numba_cuda_mlir.types.", "types."):
            if dtype.startswith(prefix):
                dtype = dtype[len(prefix) :]
                break

        type_name = _normalize_numba_mlir_type_name(dtype)
        if hasattr(numba_mlir_types, type_name):
            resolved = getattr(numba_mlir_types, type_name)
            if isinstance(resolved, numba_mlir_types.Type):
                return resolved
        raise ValueError(f"Invalid Numba-CUDA-MLIR type name: {dtype}")

    raise ValueError(f"Unrecognized dtype format: {dtype}")


_NUMBA_MLIR_DTYPE_NAMES = {
    numba_mlir_type: np_dtype.name
    for np_dtype, numba_mlir_type in _NP_DTYPE_TO_NUMBA_MLIR_TYPE.items()
}


def _normalize_common_dtype(dtype):
    """Return a Numba dtype and its backend-neutral normalized name."""

    dtype = normalize_dtype_param(dtype)
    return dtype, _NUMBA_MLIR_DTYPE_NAMES.get(dtype, str(dtype))


def _validate_common_integer_key_dtype(dtype, *, operation: str):
    """Return one normalized key dtype from the portable integer-key profile."""

    dtype, dtype_name = _normalize_common_dtype(dtype)
    validate_common_v1_integer_key_dtype_name(dtype_name, operation=operation)
    return dtype


_NO_STATIC_SENTINEL = object()


def _validate_common_merge_sort_oob_default(
    key_dtype,
    *,
    operation: str = "merge_sort_keys",
    static_value=_NO_STATIC_SENTINEL,
    runtime_dtype=None,
):
    """Validate one portable MergeSort sentinel against its inferred key dtype."""

    key_dtype, key_dtype_name = _normalize_common_dtype(key_dtype)
    validate_common_v1_integer_key_dtype_name(
        key_dtype_name,
        operation=operation,
    )

    if static_value is not _NO_STATIC_SENTINEL:
        sentinel_type = type(static_value)
        if sentinel_type is int:
            sentinel_dtype_name = "int32"
        else:
            sentinel_dtype_name = getattr(
                sentinel_type,
                "__name__",
                str(sentinel_type),
            ).lower()
    elif runtime_dtype is not None:
        try:
            _, sentinel_dtype_name = _normalize_common_dtype(runtime_dtype)
        except ValueError:
            sentinel_dtype_name = getattr(
                runtime_dtype,
                "name",
                str(runtime_dtype),
            ).lower()
    else:
        return key_dtype

    if (
        static_value is not _NO_STATIC_SENTINEL
        and isinstance(static_value, Integral)
        and not isinstance(static_value, bool)
    ):
        value = int(static_value)
        width = int(key_dtype_name.removeprefix("u").removeprefix("int"))
        if key_dtype_name.startswith("uint"):
            lower, upper = 0, (1 << width) - 1
        else:
            lower, upper = -(1 << (width - 1)), (1 << (width - 1)) - 1
        if not lower <= value <= upper:
            raise ValueError(
                f"cuda.coop.{operation} oob_default={value} is not "
                f"representable in keys dtype {key_dtype_name}"
            )

    if sentinel_dtype_name != key_dtype_name:
        raise TypeError(
            f"cuda.coop.{operation} oob_default must have the same integer "
            f"dtype as keys ({key_dtype_name}); got {sentinel_dtype_name}"
        )
    return key_dtype


def _validate_common_histogram_dtypes(sample_dtype, counter_dtype):
    """Normalize and validate the common V1 histogram dtype pair."""

    sample_dtype, sample_dtype_name = _normalize_common_dtype(sample_dtype)
    counter_dtype, counter_dtype_name = _normalize_common_dtype(counter_dtype)
    validate_common_v1_integer_value_dtype_name(
        sample_dtype_name,
        operation="histogram",
        parameter="sample",
    )
    validate_common_v1_integer_key_dtype_name(
        counter_dtype_name,
        operation="histogram",
        parameter="counter",
    )
    return sample_dtype, counter_dtype


def _validate_common_run_length_decode_dtypes(
    run_values_dtype,
    run_lengths_dtype,
):
    """Normalize and validate the common V1 Run Length Decode dtype pair."""

    run_values_dtype, run_values_dtype_name = _normalize_common_dtype(run_values_dtype)
    run_lengths_dtype, run_lengths_dtype_name = _normalize_common_dtype(
        run_lengths_dtype
    )
    validate_common_v1_integer_value_dtype_name(
        run_values_dtype_name,
        operation="run_length_decode",
        parameter="run_values",
    )
    validate_common_v1_integer_key_dtype_name(
        run_lengths_dtype_name,
        operation="run_length_decode",
        parameter="run_lengths",
    )
    return run_values_dtype, run_lengths_dtype


def _validate_common_numeric_dtype(
    dtype,
    *,
    operation: str,
    parameter: str | None = None,
):
    """Return one normalized dtype from the portable numeric profile."""

    dtype, dtype_name = _normalize_common_dtype(dtype)
    validate_common_v1_numeric_dtype_name(
        dtype_name,
        operation=operation,
        parameter=parameter,
    )
    return dtype


def _scalar_cpp_literal(value):
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(int(value))
    if isinstance(value, float):
        if np.isnan(value):
            return "NAN"
        if np.isposinf(value):
            return "INFINITY"
        if np.isneginf(value):
            return "-INFINITY"
        return repr(float(value))

    raise ValueError(
        f"Unsupported scalar literal type for compile-time binding: {type(value)}"
    )


def make_typed_cpp_literal(value, dtype):
    dtype = normalize_dtype_param(dtype)
    from ._types import numba_type_to_cpp

    cpp_type = numba_type_to_cpp(dtype)
    if cpp_type == "storage_t":
        raise ValueError(
            "Compile-time scalar literal binding does not support user-defined dtypes"
        )

    literal = _scalar_cpp_literal(value)
    return f"static_cast<{cpp_type}>({literal})"
