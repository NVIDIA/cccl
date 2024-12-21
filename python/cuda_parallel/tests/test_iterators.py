from cuda.parallel.experimental.iterators import (
    CacheModifiedInputIterator,
    ConstantIterator,
    CountingIterator,
    TransformIterator,
)
import cupy as cp
import numpy as np


def test_constant_iterator_equality():
    assert ConstantIterator(np.int32(0)) == ConstantIterator(np.int32(0))
    assert ConstantIterator(np.int32(0)) != ConstantIterator(np.int32(1))
    assert ConstantIterator(np.int32(0)) != ConstantIterator(np.int64(0))


def test_counting_iterator_equality():
    assert CountingIterator(np.int32(0)) == CountingIterator(np.int32(0))
    assert CountingIterator(np.int32(0)) != CountingIterator(np.int32(1))
    assert CountingIterator(np.int32(0)) != CountingIterator(np.int64(0))


def test_cache_modified_input_iterator_equality():
    ary1 = cp.asarray([0, 1, 2])
    ary2 = cp.asarray([3, 4, 5])
    assert CacheModifiedInputIterator(ary1, "stream") == CacheModifiedInputIterator(
        ary1, "stream"
    )
    assert CacheModifiedInputIterator(ary1, "stream") != CacheModifiedInputIterator(
        ary2, "stream"
    )


def test_equality_transform_iterator():
    def op1(x):
        return x

    def op2(x):
        return 2 * x

    def op3(x):
        return x

    it = CountingIterator(np.int32(0))
    assert TransformIterator(it, op1) == TransformIterator(it, op1)
    assert TransformIterator(it, op1) != TransformIterator(it, op2)
    assert TransformIterator(it, op1) == TransformIterator(it, op3)

    ary1 = cp.asarray([0, 1, 2])
    ary2 = cp.asarray([3, 4, 5])
    assert TransformIterator(ary1, op1) == TransformIterator(ary1, op1)
    assert TransformIterator(ary1, op1) != TransformIterator(ary1, op2)
    assert TransformIterator(ary1, op1) == TransformIterator(ary1, op3)
    assert TransformIterator(ary1, op1) != TransformIterator(ary2, op1)


def test_different_iterator_types_equality():
    assert CountingIterator(np.int32(0)) != ConstantIterator(np.int64(0))
