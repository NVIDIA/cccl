# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run-length decode signatures for block groups."""

from typing import TypeAlias, overload

from numpy import bool_ as _NumpyBool
from numpy import float16 as _NumpyFloat16
from numpy import float32 as _NumpyFloat32
from numpy import float64 as _NumpyFloat64
from numpy import int8 as _NumpyInt8
from numpy import int16 as _NumpyInt16
from numpy import int32 as _NumpyInt32
from numpy import int64 as _NumpyInt64
from numpy import uint8 as _NumpyUint8
from numpy import uint16 as _NumpyUint16
from numpy import uint32 as _NumpyUint32
from numpy import uint64 as _NumpyUint64
from typing_extensions import TypeVar

from .._typing import ThreadDataLike as _ThreadDataLike
from .._typing import _CompilerScalarLike as _CompilerScalarLike
from .._typing import _IntegerValue as _IntegerValue
from .._typing import _PortableRunLength as _PortableRunLength
from .._typing import _PortableRunValue as _PortableRunValue
from .._typing import _TraceInteger as _TraceInteger
from ._thread_group import _BlockGroup

_RunValueT = TypeVar("_RunValueT", bound=_PortableRunValue)

_RunLengthT = TypeVar("_RunLengthT", bound=_PortableRunLength)

_NumbaRunValue: TypeAlias = (
    bool
    | int
    | float
    | _NumpyBool
    | _NumpyInt8
    | _NumpyUint8
    | _NumpyInt16
    | _NumpyUint16
    | _NumpyInt32
    | _NumpyUint32
    | _NumpyInt64
    | _NumpyUint64
    | _NumpyFloat16
    | _NumpyFloat32
    | _NumpyFloat64
    | _CompilerScalarLike
)

_NumbaRunLength: TypeAlias = (
    _PortableRunLength | _NumpyInt8 | _NumpyUint8 | _NumpyInt16 | _NumpyUint16
)

_NumbaRunValueT = TypeVar("_NumbaRunValueT", bound=_NumbaRunValue)

_NumbaRunLengthT = TypeVar("_NumbaRunLengthT", bound=_NumbaRunLength)

@overload
def run_length_decode(
    group: _BlockGroup,
    run_values: _ThreadDataLike[_RunValueT],
    run_lengths: _ThreadDataLike[_RunLengthT],
    /,
    *,
    decoded_items_per_thread: _TraceInteger,
    decoded_window_offset: _IntegerValue = 0,
    relative_offsets: _ThreadDataLike[_RunLengthT] | None = None,
    total_decoded_size: _ThreadDataLike[_RunLengthT] | None = None,
    decoded_offset_dtype: object = None,
) -> _ThreadDataLike[_RunValueT]:
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
    group: _BlockGroup,
    run_values: _ThreadDataLike[_NumbaRunValueT],
    run_lengths: _ThreadDataLike[_NumbaRunLengthT],
    /,
    *,
    decoded_items_per_thread: _TraceInteger,
    decoded_window_offset: _IntegerValue = 0,
    relative_offsets: _ThreadDataLike[_NumbaRunLengthT] | None = None,
    total_decoded_size: _ThreadDataLike[_NumbaRunLengthT] | None = None,
    decoded_offset_dtype: object = None,
) -> _ThreadDataLike[_NumbaRunValueT]:
    """Decode using the broader Numba-CUDA-MLIR scalar dtype surface."""
