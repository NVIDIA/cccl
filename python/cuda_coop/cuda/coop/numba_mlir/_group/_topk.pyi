# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Describe block TopK payload types for static type checkers.

Key results retain their input item type; pair results retain each type
independently. The compiler also checks fixed extents, supported numeric
types, and count constraints. These signatures do not express which output
positions lie in the defined selected prefix.
"""

from typing_extensions import TypeVar

from ..._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    IntegralScalar,
    ThreadDataLike,
)
from .._temp_storage import TempStorage
from .._thread_group import BlockGroup

_K = TypeVar("_K", bound=CommonNumericScalar)
_V = TypeVar("_V", bound=CommonNumericScalar)

def topk_min_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_K],
    /,
    *,
    k: IntegralScalar,
    valid_items: IntegralScalar | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadDataLike[_K]: ...
def topk_min_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_K],
    values: CommonThreadDataLike[_V],
    /,
    *,
    k: IntegralScalar,
    valid_items: IntegralScalar | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[ThreadDataLike[_K], ThreadDataLike[_V]]: ...
def topk_max_keys(
    group: BlockGroup,
    keys: CommonThreadDataLike[_K],
    /,
    *,
    k: IntegralScalar,
    valid_items: IntegralScalar | None = None,
    temp_storage: TempStorage | None = None,
) -> ThreadDataLike[_K]: ...
def topk_max_pairs(
    group: BlockGroup,
    keys: CommonThreadDataLike[_K],
    values: CommonThreadDataLike[_V],
    /,
    *,
    k: IntegralScalar,
    valid_items: IntegralScalar | None = None,
    temp_storage: TempStorage | None = None,
) -> tuple[ThreadDataLike[_K], ThreadDataLike[_V]]: ...
