# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest

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


def test_transform_rai_constant():
    def constant_op(value):
        del value
        return 88

    rai = cudax_rai.TransformRAI(unary_op=constant_op, origin=0)
    assert rai.dereference(0) == 88
    assert rai.dereference(9) == 88


def test_transform_rai_counting():
    def identity_op(distance):
        return distance

    rai = cudax_rai.TransformRAI(unary_op=identity_op, origin=3)
    assert rai.dereference(0) == 3
    assert rai.dereference(9) == 12


def op_with_list_data(distance):
    permutation = [4, 2, 0, 3, 1]
    return permutation[distance % len(permutation)]


def op_with_numpy_data(distance):
    permutation = np.array([4, 2, 0, 3, 1], np.int32)
    return permutation[distance % len(permutation)]


@pytest.mark.parametrize(
    "op_with_data",
    [
        op_with_list_data,
        op_with_numpy_data,
    ],
)
def test_transform_rai_op_with_data(op_with_data):
    rai = cudax_rai.TransformRAI(unary_op=op_with_data, origin=0)
    assert rai.dereference(0) == 4
    assert rai.dereference(6) == 2
    assert rai.dereference(7) == 0
    assert rai.dereference(-2) == 3
    assert rai.dereference(-1) == 1
