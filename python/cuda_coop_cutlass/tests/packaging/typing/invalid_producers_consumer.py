# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Require producer capability, static count, and reconstruction metadata."""

from __future__ import annotations

from cutlass import Int32

import cuda.coop.cutlass as coop


class Indexable:
    def __getitem__(self, index: int, /) -> Int32:
        return Int32(index)


class Producer:
    def __cuda_coop_thread_data_load__(self) -> Indexable:
        return Indexable()


class InvalidProducer:
    def __cuda_coop_thread_data_load__(self) -> int:
        return 2


class ShapeOnly:
    shape: tuple[int, ...] = (2,)


class DtypeOnly:
    dtype: object = Int32


coop.ThreadData.load(object())  # expected-error: [call-overload]
coop.ThreadData.load(InvalidProducer())  # expected-error: [call-overload]
coop.ThreadData.load(Producer())  # expected-error: [call-overload]
coop.ThreadData.load(  # expected-error: [call-overload]
    Producer(), dtype=Int32
)
coop.ThreadData.load(  # expected-error: [call-overload]
    Producer(), items_per_thread=1.5
)
payload = coop.ThreadData.from_values(Int32(1), Int32(2))
payload.to_tensor_ssa(like=ShapeOnly())  # expected-error: [arg-type]
payload.to_tensor_ssa(like=DtypeOnly())  # expected-error: [arg-type]
payload.to_tensor_ssa(like=object())  # expected-error: [arg-type]
source: coop.ThreadDataSource[Int32]
source = Producer()  # expected-error: [assignment]
