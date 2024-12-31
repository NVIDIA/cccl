from . import _iterators
import numba


def CacheModifiedInputIterator(device_array, modifier):
    """Random Access Cache Modified Iterator that wraps a native device pointer.

    Similar to https://nvidia.github.io/cccl/cub/api/classcub_1_1CacheModifiedInputIterator.html

    Currently the only supported modifier is "stream" (LOAD_CS).
    """
    if modifier != "stream":
        raise NotImplementedError("Only stream modifier is supported")
    return _iterators.CacheModifiedPointer(
        device_array.__cuda_array_interface__["data"][0],
        numba.from_dtype(device_array.dtype),
    )


def ConstantIterator(value):
    """Returns an Iterator representing a sequence of constant values."""
    return _iterators.ConstantIterator(value)


def CountingIterator(offset):
    """Returns an Iterator representing a sequence of incrementing values."""
    return _iterators.CountingIterator(offset)


def TransformIterator(it, op):
    """Returns an Iterator representing a transformed sequence of values."""
    return _iterators.make_transform_iterator(it, op)
