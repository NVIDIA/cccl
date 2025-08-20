import numba

from ._iterators import (
    CacheModifiedPointer as _CacheModifiedPointer,
)
from ._iterators import (
    ConstantIterator as _ConstantIterator,
)
from ._iterators import (
    CountingIterator as _CountingIterator,
)
from ._iterators import (
    IteratorIOKind,
    make_reverse_iterator,
    make_transform_iterator,
)
from ._zip_iterator import make_zip_iterator


def CacheModifiedInputIterator(device_array, modifier):
    """Random Access Cache Modified Iterator that wraps a native device pointer.

    Similar to https://nvidia.github.io/cccl/cub/api/classcub_1_1CacheModifiedInputIterator.html

    Currently the only supported modifier is "stream" (LOAD_CS).

    Example:
        The code snippet below demonstrates the usage of a ``CacheModifiedInputIterator``:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_reduce_api.py
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
    return _CacheModifiedPointer(
        device_array.__cuda_array_interface__["data"][0],
        numba.from_dtype(device_array.dtype),
    )


def ConstantIterator(value):
    """Returns an Iterator representing a sequence of constant values.

    Similar to https://nvidia.github.io/cccl/thrust/api/classthrust_1_1constant__iterator.html

    Example:
        The code snippet below demonstrates the usage of a ``ConstantIterator``
        representing the sequence ``[10, 10, 10]``:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin constant-iterator
            :end-before: example-end constant-iterator

    Args:
        value: The value of every item in the sequence

    Returns:
        A ``ConstantIterator`` object initialized to ``value``
    """
    return _ConstantIterator(value)


def CountingIterator(offset):
    """Returns an Iterator representing a sequence of incrementing values.

    Similar to https://nvidia.github.io/cccl/thrust/api/classthrust_1_1counting__iterator.html

    Example:
        The code snippet below demonstrates the usage of a ``CountingIterator``
        representing the sequence ``[10, 11, 12]``:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_reduce_api.py
            :language: python
            :dedent:
            :start-after: example-begin counting-iterator
            :end-before: example-end counting-iterator

    Args:
        offset: The initial value of the sequence

    Returns:
        A ``CountingIterator`` object initialized to ``offset``
    """
    return _CountingIterator(offset)


def ReverseInputIterator(sequence):
    """Returns an input Iterator over an array in reverse.

    Similar to [std::reverse_iterator](https://en.cppreference.com/w/cpp/iterator/reverse_iterator)

    Example:
        The code snippet below demonstrates the usage of a ``ReverseInputIterator``:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin reverse-input-iterator
            :end-before: example-end reverse-input-iterator

    Args:
        sequence: The iterator or CUDA device array to be reversed

    Returns:
        A ``ReverseIterator`` object initialized with ``sequence`` to use as an input

    """
    return make_reverse_iterator(sequence, IteratorIOKind.INPUT)


def ReverseOutputIterator(sequence):
    """Returns an output Iterator over an array in reverse.

    Similar to [std::reverse_iterator](https://en.cppreference.com/w/cpp/iterator/reverse_iterator)

    Example:
        The code snippet below demonstrates the usage of a ``ReverseIterator``:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_scan_api.py
            :language: python
            :dedent:
            :start-after: example-begin reverse-output-iterator
            :end-before: example-end reverse-output-iterator

    Args:
        sequence: The iterator or CUDA device array to be reversed to use as an output

    Returns:
        A ``ReverseIterator`` object initialized with ``sequence`` to use as an output

    """
    return make_reverse_iterator(sequence, IteratorIOKind.OUTPUT)


def TransformIterator(it, op):
    """Returns an Iterator representing a transformed sequence of values.

    Similar to https://nvidia.github.io/cccl/thrust/api/classthrust_1_1transform__iterator.html

    Example:
        The code snippet below demonstrates the usage of a ``TransformIterator``
        composed with a ``CountingIterator``, transforming the sequence ``[10, 11, 12]``
        by squaring each item before reducing the output:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/test_reduce_api.py
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
    return make_transform_iterator(it, op)


def ZipIterator(*iterators):
    """Returns an Iterator representing a zipped sequence of values from N iterators.

    Similar to https://nvidia.github.io/cccl/thrust/api/classthrust_1_1zip__iterator.html

    The resulting iterator yields gpu_struct objects with fields corresponding to each input iterator.
    For 2 iterators, fields are named 'first' and 'second'. For N iterators, fields are indexed
    as field_0, field_1, ..., field_N-1.

    Example:
        The code snippet below demonstrates the usage of a ``ZipIterator``
        combining two device arrays:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/zip_iterator.py
            :language: python
            :pyobject: zip_iterator_example

    Args:
        *iterators: Variable number of iterators to zip (at least 1)

    Returns:
        A ``ZipIterator`` object that yields combined values from all input iterators
    """
    return make_zip_iterator(*iterators)
