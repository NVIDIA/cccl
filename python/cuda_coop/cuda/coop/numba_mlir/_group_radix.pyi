# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Radix rank and sort signatures for block groups."""

from typing import overload

import numpy as np
from typing_extensions import TypeVar

from .._typing import (
    IntegerValue,
    PortableIntegerKey,
    PortableNumericScalar,
    ThreadDataLike,
    TraceInteger,
)
from ._temp_storage import TempStorage
from ._thread_group import BlockGroup

_IntegerKeyT = TypeVar("_IntegerKeyT", bound=PortableIntegerKey)

_RadixValueT = TypeVar("_RadixValueT", bound=PortableNumericScalar)

@overload
def radix_sort_keys(
    group: BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> ThreadDataLike[_IntegerKeyT]:
    """Return a fresh radix-sorted block payload."""

@overload
def radix_sort_keys(
    group: BlockGroup,
    keys: _IntegerKeyT,
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> _IntegerKeyT:
    """Return one fresh radix-sorted scalar key per block thread."""

@overload
def radix_sort_pairs(
    group: BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    values: ThreadDataLike[_RadixValueT],
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> tuple[ThreadDataLike[_IntegerKeyT], ThreadDataLike[_RadixValueT]]:
    """Return fresh radix-sorted key/value payloads."""

@overload
def radix_sort_pairs(
    group: BlockGroup,
    keys: _IntegerKeyT,
    values: _RadixValueT,
    /,
    *,
    begin_bit: IntegerValue = 0,
    end_bit: IntegerValue | None = None,
    descending: bool = False,
    temp_storage: TempStorage | None = None,
    blocked_to_striped: bool = False,
) -> tuple[_IntegerKeyT, _RadixValueT]:
    """Return one fresh radix-sorted scalar pair per block thread."""

@overload
def radix_rank(
    group: BlockGroup,
    keys: ThreadDataLike[_IntegerKeyT],
    /,
    *,
    begin_bit: TraceInteger = 0,
    end_bit: TraceInteger | None = None,
    radix_bits: TraceInteger | None = None,
    descending: bool = False,
    exclusive_digit_prefix: ThreadDataLike[int]
    | ThreadDataLike[np.int32]
    | None = None,
) -> ThreadDataLike[int]:
    """Return fresh signed 32-bit ranks for one radix digit."""

@overload
def radix_rank(
    group: BlockGroup,
    keys: _IntegerKeyT,
    /,
    *,
    begin_bit: TraceInteger = 0,
    end_bit: TraceInteger | None = None,
    radix_bits: TraceInteger | None = None,
    descending: bool = False,
    exclusive_digit_prefix: ThreadDataLike[int]
    | ThreadDataLike[np.int32]
    | None = None,
) -> int:
    """Return one signed 32-bit radix rank per block thread."""
