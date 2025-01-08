import cupy as cp
import numpy as np

from cuda.parallel.experimental.iterators import (
    CacheModifiedInputIterator,
    ConstantIterator,
    CountingIterator,
    TransformIterator,
)


def test_constant_iterator_equality():
    it1 = ConstantIterator(np.int32(0))
    it2 = ConstantIterator(np.int32(0))
    it3 = ConstantIterator(np.int32(1))
    it4 = ConstantIterator(np.int64(9))

    assert it1.kind == it2.kind == it3.kind
    assert it1.kind != it4.kind


def test_counting_iterator_equality():
    it1 = CountingIterator(np.int32(0))
    it2 = CountingIterator(np.int32(0))
    it3 = CountingIterator(np.int32(1))
    it4 = CountingIterator(np.int64(9))

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
    it1 = TransformIterator(it, op1)
    it2 = TransformIterator(it, op1)
    it3 = TransformIterator(it, op3)

    assert it1.kind == it2.kind == it3.kind

    ary1 = cp.asarray([0, 1, 2])
    ary2 = cp.asarray([3, 4, 5])
    it4 = TransformIterator(ary1, op1)
    it5 = TransformIterator(ary1, op1)
    it6 = TransformIterator(ary1, op2)
    it7 = TransformIterator(ary1, op3)
    it8 = TransformIterator(ary2, op1)

    assert it4.kind == it5.kind == it7.kind == it8.kind
    assert it4.kind != it6.kind
