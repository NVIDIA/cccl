# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Radix rank and sort signatures for block groups."""

from typing import overload

from numpy import int32 as _NumpyInt32
from typing_extensions import TypeVar

from .._typing import ThreadDataLike as _ThreadDataLike
from .._typing import _IntegerValue as _IntegerValue
from .._typing import _PortableIntegerKey as _PortableIntegerKey
from .._typing import _PortableNumericScalar as _PortableNumericScalar
from .._typing import _TraceInteger as _TraceInteger
from ._temp_storage import TempStorage
from ._thread_group import _BlockGroup

_IntegerKeyT = TypeVar("_IntegerKeyT", bound=_PortableIntegerKey)

_RadixValueT = TypeVar("_RadixValueT", bound=_PortableNumericScalar)

@overload
def radix_sort_keys(
    group: _BlockGroup,
    keys: _ThreadDataLike[_IntegerKeyT],
    /,
    *,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> _ThreadDataLike[_IntegerKeyT]:
    """Return a fresh radix-sorted block payload."""

@overload
def radix_sort_keys(
    group: _BlockGroup,
    keys: _IntegerKeyT,
    /,
    *,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> _IntegerKeyT:
    """Return one fresh radix-sorted scalar key per block thread."""

@overload
def radix_sort_pairs(
    group: _BlockGroup,
    keys: _ThreadDataLike[_IntegerKeyT],
    values: _ThreadDataLike[_RadixValueT],
    /,
    *,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> tuple[_ThreadDataLike[_IntegerKeyT], _ThreadDataLike[_RadixValueT]]:
    """Return fresh radix-sorted key/value payloads."""

@overload
def radix_sort_pairs(
    group: _BlockGroup,
    keys: _IntegerKeyT,
    values: _RadixValueT,
    /,
    *,
    begin_bit: _IntegerValue = 0,
    end_bit: _IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> tuple[_IntegerKeyT, _RadixValueT]:
    """Return one fresh radix-sorted scalar pair per block thread."""

@overload
def radix_rank(
    group: _BlockGroup,
    keys: _ThreadDataLike[_IntegerKeyT],
    /,
    *,
    begin_bit: _TraceInteger = 0,
    end_bit: _TraceInteger | None = None,
    radix_bits: _TraceInteger | None = None,
    descending: bool = False,
    exclusive_digit_prefix: _ThreadDataLike[int]
    | _ThreadDataLike[_NumpyInt32]
    | None = None,
) -> _ThreadDataLike[int]:
    """Return fresh signed 32-bit ranks for one radix digit."""

@overload
def radix_rank(
    group: _BlockGroup,
    keys: _IntegerKeyT,
    /,
    *,
    begin_bit: _TraceInteger = 0,
    end_bit: _TraceInteger | None = None,
    radix_bits: _TraceInteger | None = None,
    descending: bool = False,
    exclusive_digit_prefix: _ThreadDataLike[int]
    | _ThreadDataLike[_NumpyInt32]
    | None = None,
) -> int:
    """Return one signed 32-bit radix rank per block thread."""
