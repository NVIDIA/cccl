# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Same-dtype NumPy and CUTLASS scalar seed contracts.

CUTLASS scalar annotations do not encode the canonical dtype structurally,
so mixed NumPy/compiler seeds need one nominal pairing per supported dtype.
The compiler protocols also require scalar metadata, so a missing optional
CUTLASS import cannot turn a seed or input contract into unrestricted Any.
"""

from typing import Protocol, TypeAlias, TypeVar

import numpy

# CUTLASS is optional for consumers of the common API.
from cutlass import (  # type: ignore[import-not-found, unused-ignore]
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
)

from cuda.coop._typing import CompilerScalarLike, _ExactScalar

# Missing CUTLASS types become Any only inside these guarded protocols.
# The per-class ignores cover that optional import, not consumer arguments.
_T = TypeVar("_T")

class _CompilerIdentity(CompilerScalarLike, _ExactScalar[_T], Protocol[_T]): ...
class _CutlassInt8(_CompilerIdentity[Int8], Protocol):  # type: ignore[no-any-unimported, unused-ignore]
    ...

_SeedInt8: TypeAlias = _ExactScalar[numpy.int8]
_NumpyInt8T = TypeVar("_NumpyInt8T", bound=numpy.int8)
_CompilerInt8T = TypeVar("_CompilerInt8T", bound=_CutlassInt8)

class _CutlassUint8(_CompilerIdentity[Uint8], Protocol):  # type: ignore[no-any-unimported, unused-ignore]
    ...

_SeedUint8: TypeAlias = _ExactScalar[numpy.uint8]
_NumpyUint8T = TypeVar("_NumpyUint8T", bound=numpy.uint8)
_CompilerUint8T = TypeVar("_CompilerUint8T", bound=_CutlassUint8)

class _CutlassInt16(_CompilerIdentity[Int16], Protocol):  # type: ignore[no-any-unimported, unused-ignore]
    ...

_SeedInt16: TypeAlias = _ExactScalar[numpy.int16]
_NumpyInt16T = TypeVar("_NumpyInt16T", bound=numpy.int16)
_CompilerInt16T = TypeVar("_CompilerInt16T", bound=_CutlassInt16)

class _CutlassUint16(_CompilerIdentity[Uint16], Protocol):  # type: ignore[no-any-unimported, unused-ignore]
    ...

_SeedUint16: TypeAlias = _ExactScalar[numpy.uint16]
_NumpyUint16T = TypeVar("_NumpyUint16T", bound=numpy.uint16)
_CompilerUint16T = TypeVar("_CompilerUint16T", bound=_CutlassUint16)

class _CutlassInt32(_CompilerIdentity[Int32], Protocol):  # type: ignore[no-any-unimported, unused-ignore]
    ...

_SeedInt32: TypeAlias = _ExactScalar[numpy.int32]
_NumpyInt32T = TypeVar("_NumpyInt32T", bound=numpy.int32)
_CompilerInt32T = TypeVar("_CompilerInt32T", bound=_CutlassInt32)

class _CutlassUint32(_CompilerIdentity[Uint32], Protocol):  # type: ignore[no-any-unimported, unused-ignore]
    ...

_SeedUint32: TypeAlias = _ExactScalar[numpy.uint32]
_NumpyUint32T = TypeVar("_NumpyUint32T", bound=numpy.uint32)
_CompilerUint32T = TypeVar("_CompilerUint32T", bound=_CutlassUint32)

class _CutlassInt64(_CompilerIdentity[Int64], Protocol):  # type: ignore[no-any-unimported, unused-ignore]
    ...

_SeedInt64: TypeAlias = _ExactScalar[numpy.int64]
_NumpyInt64T = TypeVar("_NumpyInt64T", bound=numpy.int64)
_CompilerInt64T = TypeVar("_CompilerInt64T", bound=_CutlassInt64)

class _CutlassUint64(_CompilerIdentity[Uint64], Protocol):  # type: ignore[no-any-unimported, unused-ignore]
    ...

_SeedUint64: TypeAlias = _ExactScalar[numpy.uint64]
_NumpyUint64T = TypeVar("_NumpyUint64T", bound=numpy.uint64)
_CompilerUint64T = TypeVar("_CompilerUint64T", bound=_CutlassUint64)

class _CutlassFloat32(_CompilerIdentity[Float32], Protocol):  # type: ignore[no-any-unimported, unused-ignore]
    ...

_SeedFloat32: TypeAlias = _ExactScalar[numpy.float32]
_NumpyFloat32T = TypeVar("_NumpyFloat32T", bound=numpy.float32)
_CompilerFloat32T = TypeVar("_CompilerFloat32T", bound=_CutlassFloat32)

class _CutlassFloat64(_CompilerIdentity[Float64], Protocol):  # type: ignore[no-any-unimported, unused-ignore]
    ...

_SeedFloat64: TypeAlias = _ExactScalar[numpy.float64]
_NumpyFloat64T = TypeVar("_NumpyFloat64T", bound=numpy.float64)
_CompilerFloat64T = TypeVar("_CompilerFloat64T", bound=_CutlassFloat64)
