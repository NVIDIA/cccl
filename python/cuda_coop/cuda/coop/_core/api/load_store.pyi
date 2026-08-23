# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Typing contract for portable cooperative load and store."""

from typing import overload

from typing_extensions import TypeVar

from cuda.coop._typing import TempStorageLike as TempStorageLike
from cuda.coop._typing import ThreadDataLike as ThreadDataLike
from cuda.coop._typing import _BlockLoadStoreAlgorithm as _BlockLoadStoreAlgorithm
from cuda.coop._typing import _IntegerValue as _IntegerValue
from cuda.coop._typing import _PortableNumericScalar as _PortableNumericScalar
from cuda.coop._typing import _PortableThreadDataLike as _PortableThreadDataLike
from cuda.coop._typing import _ValidItems as _ValidItems
from cuda.coop._typing import _WarpLoadStoreAlgorithm as _WarpLoadStoreAlgorithm

from .thread_group import _BlockGroup, _WarpGroup

_PortableNumericT = TypeVar("_PortableNumericT", bound=_PortableNumericScalar)

@overload
def load(
    group: _BlockGroup,
    source: object,
    output: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: _BlockLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    oob_default: None = None,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Populate and return ``output`` with one cooperative block tile."""

@overload
def load(
    group: _BlockGroup,
    source: object,
    output: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: _BlockLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems,
    oob_default: _PortableNumericT,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Populate a partial block tile and fill invalid items."""

@overload
def load(
    group: _WarpGroup,
    source: object,
    output: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: _WarpLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    oob_default: None = None,
    offset: _IntegerValue | None = None,
    temp_storage: None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Populate and return ``output`` with a physical- or logical-warp tile."""

@overload
def load(
    group: _WarpGroup,
    source: object,
    output: ThreadDataLike[_PortableNumericT],
    /,
    *,
    algorithm: _WarpLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems,
    oob_default: _PortableNumericT,
    offset: _IntegerValue | None = None,
    temp_storage: None = None,
) -> ThreadDataLike[_PortableNumericT]:
    """Populate a partial physical- or logical-warp tile and fill invalid items."""

@overload
def store(
    group: _BlockGroup,
    destination: object,
    value: _PortableNumericScalar | _PortableThreadDataLike,
    /,
    *,
    algorithm: _BlockLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    offset: _IntegerValue | None = None,
    temp_storage: TempStorageLike | None = None,
) -> None:
    """Store one scalar or per-thread payload cooperatively across a block."""

@overload
def store(
    group: _WarpGroup,
    destination: object,
    value: _PortableNumericScalar | _PortableThreadDataLike,
    /,
    *,
    algorithm: _WarpLoadStoreAlgorithm = "direct",
    valid_items: _ValidItems | None = None,
    offset: _IntegerValue | None = None,
    temp_storage: None = None,
) -> None:
    """Store one scalar or per-thread payload across a physical or logical warp."""
