from . import _iterators


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
