from . import _iterators


def cache_load_modifier(device_array, ntype, modifier):
    """Python fascade for Random Access Iterator that wraps a native device pointer.

    Similar to https://nvidia.github.io/cccl/cub/api/classcub_1_1CacheModifiedInputIterator.html

    Currently the only supported modifier is "stream" (LOAD_CS).
    """
    if modifier != "stream":
        raise NotImplementedError("Only stream modifier is supported")
    return _iterators.CacheModifiedPointer(
        device_array.device_ctypes_pointer.value, ntype
    )


def repeat(value, ntype):
    """Python fascade (similar to itertools.repeat) for C++ Random Access ConstantIterator."""
    return _iterators.ConstantIterator(value, ntype)


def count(offset, ntype):
    """Python fascade (similar to itertools.count) for C++ Random Access ConstantIterator."""
    return _iterators.CountingIterator(offset, ntype)


# Knowingly shadowing a built-in function.
def map(op, it, op_return_ntype):
    """Python fascade (similar to built-in map) mimicking a C++ Random Access TransformIterator."""
    return _iterators.cumap(op, it, op_return_ntype)
