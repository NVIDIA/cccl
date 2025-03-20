import numba

from . import _cy_iterators as _iterators


def CacheModifiedInputIterator(device_array, modifier, prefix=""):
    """Random Access Cache Modified Iterator that wraps a native device pointer.

    Similar to https://nvidia.github.io/cccl/cub/api/classcub_1_1CacheModifiedInputIterator.html

    Currently the only supported modifier is "stream" (LOAD_CS).

    Example:
        The code snippet below demonstrates the usage of a ``CacheModifiedInputIterator``:

        .. literalinclude:: ../../python/cuda_parallel/tests/test_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin cache-iterator
            :end-before: example-end cache-iterator

    Args:
        device_array: CUDA device array storing the input sequence of data items
        modifier: The PTX cache load modifier
        prefix: An optional prefix added to the iterator's methods to prevent name collisions.

    Returns:
        A ``CacheModifiedInputIterator`` object initialized with ``device_array``
    """
    if modifier != "stream":
        raise NotImplementedError("Only stream modifier is supported")
    return _iterators.CacheModifiedPointer(
        device_array.__cuda_array_interface__["data"][0],
        numba.from_dtype(device_array.dtype),
        prefix,
    )


def ConstantIterator(value):
    """Returns an Iterator representing a sequence of constant values.

    Similar to https://nvidia.github.io/cccl/thrust/api/classthrust_1_1constant__iterator.html

    Example:
        The code snippet below demonstrates the usage of a ``ConstantIterator``
        representing the sequence ``[10, 10, 10]``:

        .. literalinclude:: ../../python/cuda_parallel/tests/test_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin constant-iterator
            :end-before: example-end constant-iterator

    Args:
        value: The value of every item in the sequence

    Returns:
        A ``ConstantIterator`` object initialized to ``value``
    """
    return _iterators.ConstantIterator(value)


def CountingIterator(offset):
    """Returns an Iterator representing a sequence of incrementing values.

    Similar to https://nvidia.github.io/cccl/thrust/api/classthrust_1_1counting__iterator.html

    Example:
        The code snippet below demonstrates the usage of a ``CountingIterator``
        representing the sequence ``[10, 11, 12]``:

        .. literalinclude:: ../../python/cuda_parallel/tests/test_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin counting-iterator
            :end-before: example-end counting-iterator

    Args:
        offset: The initial value of the sequence

    Returns:
        A ``CountingIterator`` object initialized to ``offset``
    """
    return _iterators.CountingIterator(offset)


def TransformIterator(it, op):
    """Returns an Iterator representing a transformed sequence of values.

    Similar to https://nvidia.github.io/cccl/thrust/api/classthrust_1_1transform__iterator.html

    Example:
        The code snippet below demonstrates the usage of a ``TransformIterator``
        composed with a ``CountingIterator``, transforming the sequence ``[10, 11, 12]``
        by squaring each item before reducing the output:

        .. literalinclude:: ../../python/cuda_parallel/tests/test_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin transform-iterator
            :end-before: example-end transform-iterator

    Args:
        it: The iterator object to be transformed
        op: The transform operation

    Returns:
        A ``TransformIterator`` object to transform the items in ``it`` using ``op``
    """
    return _iterators.make_transform_iterator(it, op)
