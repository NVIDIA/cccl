# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
import pytest
from _utils.device_array import DeviceArray

from cuda.compute import (
    CountingIterator,
    OpKind,
    TransformIterator,
    TransformOutputIterator,
    reduce_into,
    types,
)
from cuda.compute.types import signature_from_annotations

T = TypeVar("T")


def square_any(x: Any) -> Any:
    return x * x


def square_object(x: object) -> object:
    return x * x


def square_numpy_object(x: np.object_) -> np.object_:
    return x * x


def square_typevar(x: T) -> T:
    return x * x


def square_float(x: np.float64) -> np.float64:
    return x * x


@pytest.mark.parametrize(
    "op", [square_any, square_object, square_numpy_object, square_typevar]
)
def test_nonconcrete_annotations_are_absent(op):
    assert signature_from_annotations(op) == ([], None)
    iterator = TransformIterator(CountingIterator(np.float64(0)), op)
    assert iterator.value_type == types.float64


def test_concrete_annotation_is_preserved():
    assert signature_from_annotations(square_float) == ([types.float64], types.float64)


def test_output_iterator_requires_concrete_input_type():
    underlying = CountingIterator(np.float64(0))
    with pytest.raises(ValueError, match="exactly one argument with type annotation"):
        TransformOutputIterator(underlying, square_any)
    iterator = TransformOutputIterator(
        underlying, square_any, output_value_type=types.float64
    )
    assert iterator.value_type == types.float64


def test_transform_iterator_any_reduction():
    output = DeviceArray.empty(1, np.float64)
    reduce_into(
        d_in=TransformIterator(CountingIterator(np.float64(0)), square_any),
        d_out=output,
        num_items=8,
        op=OpKind.PLUS,
        h_init=np.array([0], dtype=np.float64),
    )
    assert output.copy_to_host()[0] == 140.0
