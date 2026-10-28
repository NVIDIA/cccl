# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Preserve producer item types and metadata-only tensor reconstruction."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

import numpy as np
from cutlass import Int32
from typing_extensions import assert_type

import cuda.coop.cutlass as coop
from cuda.coop.cutlass._thread_data import CutlassTensorSSASample

_ItemT = TypeVar("_ItemT")


class Indexable(Generic[_ItemT]):
    """An explicitly sized register payload."""

    def __init__(self, value: _ItemT) -> None:
        self.value = value

    def __getitem__(self, index: int, /) -> _ItemT:
        return self.value


class Shaped(Indexable[_ItemT]):
    shape: tuple[int, ...] = (2,)


class Counted(Indexable[_ItemT]):
    def numel(self) -> int:
        return 2


class ShapedProducer(Generic[_ItemT]):
    def __init__(self, value: _ItemT) -> None:
        self.value = value

    def __cuda_coop_thread_data_load__(self) -> Shaped[_ItemT]:
        return Shaped(self.value)


class CountedProducer(Generic[_ItemT]):
    def __init__(self, value: _ItemT) -> None:
        self.value = value

    def __cuda_coop_thread_data_load__(self) -> Counted[_ItemT]:
        return Counted(self.value)


class IndexableProducer(Generic[_ItemT]):
    def __init__(self, value: _ItemT) -> None:
        self.value = value

    def __cuda_coop_thread_data_load__(self) -> Indexable[_ItemT]:
        return Indexable(self.value)


class PayloadProducer:
    def __cuda_coop_thread_data_load__(self) -> coop.ThreadData[Int32]:
        return coop.ThreadData.from_values(Int32(1), Int32(2))


class Metadata:
    shape: tuple[int, ...] = (1, 2)
    dtype: object = Int32


class CombinedProducer(ShapedProducer[Int32], Metadata):
    pass


class UserScalar:
    """Producer protocols preserve user-defined scalar types."""


def check_producers(opaque_dtype: object) -> None:
    shaped = ShapedProducer(Int32(1))
    indexed = IndexableProducer(Int32(1))
    assert_type(coop.ThreadData.load(shaped), coop.ThreadData[Int32])
    assert_type(
        coop.ThreadData.load(CountedProducer(np.float64(1))),
        coop.ThreadData[np.float64],
    )
    assert_type(
        coop.ThreadData.load(indexed, items_per_thread=2),
        coop.ThreadData[Int32],
    )
    assert_type(coop.ThreadData.load(PayloadProducer()), coop.ThreadData[Int32])
    assert_type(
        coop.ThreadData.load(shaped, dtype=np.float32),
        coop.ThreadData[np.float32],
    )
    assert_type(
        coop.ThreadData.load(indexed, items_per_thread=2, dtype=np.float64),
        coop.ThreadData[np.float64],
    )
    assert_type(
        coop.ThreadData.load(shaped, dtype=opaque_dtype), coop.ThreadData[Any]
    )
    assert_type(
        coop.ThreadData.load(indexed, items_per_thread=2, dtype=opaque_dtype),
        coop.ThreadData[Any],
    )
    assert_type(
        coop.ThreadData.load(ShapedProducer(UserScalar())),
        coop.ThreadData[UserScalar],
    )
    assert_type(
        coop.ThreadData.load(shaped, dtype=UserScalar),
        coop.ThreadData[UserScalar],
    )
    source: coop.ThreadDataLoadSource[Int32] = shaped
    assert_type(coop.ThreadData.load(source), coop.ThreadData[Int32])
    combined: coop.ThreadDataSource[Int32] = CombinedProducer(Int32(1))
    metadata: coop.ThreadDataTensorMetadata = Metadata()
    payload = coop.ThreadData.load(combined)
    assert_type(payload, coop.ThreadData[Int32])
    assert_type(payload.to_tensor_ssa(like=combined), CutlassTensorSSASample)
    assert_type(
        payload.to_tensor_ssa(like=metadata, dtype=Int32, shape=(2,)),
        CutlassTensorSSASample,
    )
