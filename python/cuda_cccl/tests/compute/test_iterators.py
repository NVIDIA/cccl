import cupy as cp
import numba.cuda
import numpy as np
import pytest

from cuda.compute._utils.protocols import (
    compute_c_contiguous_strides_in_bytes,
)
from cuda.compute.iterators import (
    CacheModifiedInputIterator,
    ConstantIterator,
    CountingIterator,
    ReverseIterator,
    TransformIterator,
)


def test_constant_iterator_equality():
    it1 = ConstantIterator(np.int32(0))
    it2 = ConstantIterator(np.int32(0))
    it3 = ConstantIterator(np.int32(1))
    it4 = ConstantIterator(np.int64(0))

    assert it1.kind == it2.kind == it3.kind
    assert it1.kind != it4.kind


def test_counting_iterator_equality():
    it1 = CountingIterator(np.int32(0))
    it2 = CountingIterator(np.int32(0))
    it3 = CountingIterator(np.int32(1))
    it4 = CountingIterator(np.int64(0))

    assert it1.kind == it2.kind == it3.kind
    assert it1.kind != it4.kind


def test_cache_modified_input_iterator_equality():
    ary1 = cp.asarray([0, 1, 2], dtype="int32")
    ary2 = cp.asarray([3, 4, 5], dtype="int32")
    ary3 = cp.asarray([0, 1, 2], dtype="int64")

    it1 = CacheModifiedInputIterator(ary1, "stream")
    it2 = CacheModifiedInputIterator(ary1, "stream")
    it3 = CacheModifiedInputIterator(ary2, "stream")
    it4 = CacheModifiedInputIterator(ary3, "stream")

    assert it1.kind == it2.kind == it3.kind
    assert it1.kind != it4.kind


def test_equality_transform_iterator():
    def op1(x):
        return x

    def op2(x):
        return 2 * x

    def op3(x):
        return x

    it = CountingIterator(np.int32(0))
    it = CountingIterator(np.int32(1))
    it1 = TransformIterator(it, op1)
    it2 = TransformIterator(it, op1)
    it3 = TransformIterator(it, op3)

    assert it1.kind == it2.kind
    # op3 has a different name than op1, so should have a different kind
    assert it1.kind != it3.kind

    ary1 = cp.asarray([0, 1, 2])
    ary2 = cp.asarray([3, 4, 5])
    it4 = TransformIterator(ary1, op1)
    it5 = TransformIterator(ary1, op1)
    it6 = TransformIterator(ary1, op2)
    it7 = TransformIterator(ary1, op3)
    it8 = TransformIterator(ary2, op1)

    assert it4.kind == it5.kind == it8.kind
    # op2 has different bytecode, so should have a different kind
    assert it4.kind != it6.kind
    # op3 has a different name than op1, so should have a different kind
    assert it4.kind != it7.kind


@pytest.fixture(
    params=[
        # Each tuple is (shape, layout, array_type)
        ((5,), "C", "cupy"),
        ((5,), "F", "cupy"),
        ((5,), "C", "numba"),
        ((5,), "F", "numba"),
        ((4, 3), "C", "cupy"),
        ((4, 3), "F", "cupy"),
        ((4, 3), "C", "numba"),
        ((4, 3), "F", "numba"),
        ((3, 4, 2), "C", "cupy"),
        ((3, 4, 2), "F", "cupy"),
        ((3, 4, 2), "C", "numba"),
        ((3, 4, 2), "F", "numba"),
    ],
    ids=lambda param: f"{param[2]}_{param[1]}_{len(param[0])}D",
)
def reverse_iterator_array(request):
    shape, layout, array_type = request.param

    # Create base numpy array
    base_array = np.arange(np.prod(shape))
    base_array[-1] = -999
    base_array = base_array.reshape(shape)
    if layout == "F":
        base_array = np.asfortranarray(base_array)

    if array_type == "cupy":
        array = cp.array(base_array)
    else:
        array = numba.cuda.to_device(base_array)

    return array


def test_reverse_iterator(reverse_iterator_array):
    it = ReverseIterator(reverse_iterator_array)

    # Create array of size 1 from memory pointer of last element
    arr = cp.ndarray(
        shape=(1,),
        dtype=reverse_iterator_array.dtype,
        memptr=cp.cuda.MemoryPointer(
            cp.cuda.UnownedMemory(it.cvalue.value, 0, None), 0
        ),
    )

    assert -999 == arr[0]


def test_reverse_input_iterator_equality():
    ary1 = cp.asarray([0, 1, 2], dtype="int32")
    ary2 = cp.asarray([3, 4, 5], dtype="int32")
    ary3 = cp.asarray([0, 1, 2], dtype="int64")

    it1 = ReverseIterator(ary1)
    it2 = ReverseIterator(ary1)
    it3 = ReverseIterator(ary2)
    it4 = ReverseIterator(ary3)

    assert it1.kind == it2.kind == it3.kind
    assert it1.kind != it4.kind


def test_reverse_output_iterator_equality():
    ary1 = cp.asarray([0, 1, 2], dtype="int32")
    ary2 = cp.asarray([3, 4, 5], dtype="int32")
    ary3 = cp.asarray([0, 1, 2], dtype="int64")

    it1 = ReverseIterator(ary1)
    it2 = ReverseIterator(ary1)
    it3 = ReverseIterator(ary2)
    it4 = ReverseIterator(ary3)

    assert it1.kind == it2.kind == it3.kind
    assert it1.kind != it4.kind


@pytest.mark.parametrize(
    "shape, itemsize, expected",
    [
        # Basic 1D
        ((5,), 4, (4,)),
        ((10,), 1, (1,)),
        # Basic 2D
        ((2, 3), 4, (12, 4)),
        ((3, 2), 8, (16, 8)),
        # Basic 3D
        ((2, 3, 4), 1, (12, 4, 1)),
        ((2, 3, 4), 2, (24, 8, 2)),
        # Scalars (0D array)
        ((), 4, ()),
        # Shape with a zero-length dimension
        ((0, 3), 4, (12, 4)),
        ((3, 0), 4, (0, 4)),
    ],
)
def test_compute_c_contiguous_strides_in_bytes(shape, itemsize, expected):
    result = compute_c_contiguous_strides_in_bytes(shape, itemsize)
    assert result == expected


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((2, 3), np.int32),
        ((4, 5, 6), np.float64),
        ((10,), np.uint8),
        ((1,), np.float16),
    ],
)
def test_matches_numpy_strides_for_c_contiguous_arrays(shape, dtype):
    arr = np.zeros(shape, dtype=dtype, order="C")
    expected = arr.strides
    result = compute_c_contiguous_strides_in_bytes(shape, dtype().itemsize)
    assert result == expected
