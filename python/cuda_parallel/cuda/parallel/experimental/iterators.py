from . import _iterators


def CacheModifiedInputIterator(device_array, value_type, modifier):
    """Python fascade for Random Access Cache Modified Iterator that wraps a native device pointer.

    Similar to https://nvidia.github.io/cccl/cub/api/classcub_1_1CacheModifiedInputIterator.html

    Currently the only supported modifier is "stream" (LOAD_CS).
    """
    if modifier != "stream":
        raise NotImplementedError("Only stream modifier is supported")
    return _iterators.CacheModifiedPointer(
        device_array.__cuda_array_interface__["data"][0],
        _iterators.numba_type_from_any(value_type),
    )


def ConstantIterator(value, value_type):
    """Python fascade (similar to itertools.repeat) for C++ Random Access ConstantIterator."""
    return _iterators.ConstantIterator(
        value, _iterators.numba_type_from_any(value_type)
    )


def CountingIterator(offset, value_type):
    """Python fascade (similar to itertools.count) for C++ Random Access CountingIterator."""
    return _iterators.CountingIterator(
        offset, _iterators.numba_type_from_any(value_type)
    )


def TransformIterator(op, it, op_return_value_type):
    """Python fascade (similar to built-in map) mimicking a C++ Random Access TransformIterator."""
    return _iterators.TransformIterator(
        op, it, _iterators.numba_type_from_any(op_return_value_type)
    )
