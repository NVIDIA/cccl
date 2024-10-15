# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from cuda.parallel.experimental import random_access_iterators as cudax_rai


def test_constant_rai():
    rai = cudax_rai.ConstantRAI(value=42)
    assert rai.advance(0) is rai
    assert rai.dereference(0) == 42


def test_counting_rai():
    rai = cudax_rai.CountingRAI(origin=24)
    assert rai.advance(0) is not rai
    assert rai.advance(13).origin == 37
    assert rai.origin == 24
    assert rai.dereference(0) == 24
    assert rai.dereference(14) == 38


def test_transform_rai():
    def some_shift(value):
        return value + 15

    rai = cudax_rai.TransformRAI(unary_op=some_shift, origin=7)
    assert rai.advance(0) is not rai
    assert rai.advance(3).origin == 10
    assert rai.origin == 7
    assert rai.dereference(0) == 22
    assert rai.dereference(4) == 26
