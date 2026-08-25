# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run-length decode signatures for block groups."""

from typing import TypeAlias, overload

import numpy as np
from typing_extensions import TypeVar

from .._typing import (
    CompilerScalarLike,
    IntegerValue,
    PortableRunLength,
    PortableRunValue,
    ThreadDataLike,
    TraceInteger,
)
from ._thread_group import BlockGroup

_RunValueT = TypeVar("_RunValueT", bound=PortableRunValue)

_RunLengthT = TypeVar("_RunLengthT", bound=PortableRunLength)

_NumbaRunValue: TypeAlias = (
    bool
    | int
    | float
    | np.bool_
    | np.int8
    | np.uint8
    | np.int16
    | np.uint16
    | np.int32
    | np.uint32
    | np.int64
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
    | CompilerScalarLike
)

_NumbaRunLength: TypeAlias = (
    PortableRunLength | np.int8 | np.uint8 | np.int16 | np.uint16
)

_NumbaRunValueT = TypeVar("_NumbaRunValueT", bound=_NumbaRunValue)

_NumbaRunLengthT = TypeVar("_NumbaRunLengthT", bound=_NumbaRunLength)

@overload
def run_length_decode(
    group: BlockGroup,
    run_values: ThreadDataLike[_RunValueT],
    run_lengths: ThreadDataLike[_RunLengthT],
    /,
    *,
    decoded_items_per_thread: TraceInteger,
    decoded_window_offset: IntegerValue = 0,
    relative_offsets: ThreadDataLike[_RunLengthT] | None = None,
    total_decoded_size: ThreadDataLike[_RunLengthT] | None = None,
    decoded_offset_dtype: object = None,
) -> ThreadDataLike[_RunValueT]:
    """Decode a blockwise window with optional side outputs.

    Inputs have matching fixed extents and use blocked run ownership. The
    decoded extent is positive and static. The uniform window offset is
    nonnegative and representable in the run-length dtype; dynamic callers
    guarantee its range. Side outputs use that same dtype. Actual runs have
    positive lengths followed only by an optional trailing zero-padding
    suffix, and their positive total is representable in the length dtype.
    The decoded result is fresh, inputs are unchanged, and positions past the
    total decode to zero.
    """

@overload
def run_length_decode(
    group: BlockGroup,
    run_values: ThreadDataLike[_NumbaRunValueT],
    run_lengths: ThreadDataLike[_NumbaRunLengthT],
    /,
    *,
    decoded_items_per_thread: TraceInteger,
    decoded_window_offset: IntegerValue = 0,
    relative_offsets: ThreadDataLike[_NumbaRunLengthT] | None = None,
    total_decoded_size: ThreadDataLike[_NumbaRunLengthT] | None = None,
    decoded_offset_dtype: object = None,
) -> ThreadDataLike[_NumbaRunValueT]:
    """Decode using the broader Numba-CUDA-MLIR scalar dtype surface."""
