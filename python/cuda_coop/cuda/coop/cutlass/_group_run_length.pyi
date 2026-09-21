# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run Length Decode signatures for qualified CuTe payloads."""

from typing import Any, TypeAlias, overload

from cutlass import Uint32
from typing_extensions import TypeVar

from .._core.api.thread_group import BlockGroup
from .._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    IntegralScalar,
    TempStorageLike,
)
from ._thread_data import CutlassTensorSample, CutlassTensorSSASample, ThreadData

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)
_LengthT = TypeVar("_LengthT", bound=IntegralScalar)
_RegisterPayload: TypeAlias = CutlassTensorSample | CutlassTensorSSASample

@overload
def run_length_decode(
    group: BlockGroup,
    run_values: CommonThreadDataLike[_ItemT],
    run_lengths: CommonThreadDataLike[_LengthT] | _RegisterPayload,
    /,
    *,
    decoded_items_per_thread: int,
    decoded_window_offset: IntegralScalar = 0,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[_ItemT]: ...
@overload
def run_length_decode(
    group: BlockGroup,
    run_values: _RegisterPayload,
    run_lengths: CommonThreadDataLike[_LengthT] | _RegisterPayload,
    /,
    *,
    decoded_items_per_thread: int,
    decoded_window_offset: IntegralScalar = 0,
    temp_storage: TempStorageLike | None = None,
) -> ThreadData[Any]: ...
def run_length_decode_into(
    group: BlockGroup,
    run_values: CommonThreadDataLike[_ItemT] | _RegisterPayload,
    run_lengths: CommonThreadDataLike[_LengthT] | _RegisterPayload,
    destination: CutlassTensorSample,
    /,
    *,
    decoded_items_per_thread: int,
    destination_offset: IntegralScalar = 0,
    temp_storage: TempStorageLike | None = None,
) -> Uint32: ...
