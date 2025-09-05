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

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/cache_modified_iterator_basic.py
            :language: python
            :start-after: # example-begin


    Args:
        device_array: CUDA device array storing the input sequence of data items
        modifier: The PTX cache load modifier

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
        representing a sequence of constant values:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/constant_iterator_basic.py
            :language: python
            :start-after: # example-begin


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

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/counting_iterator_basic.py
            :language: python
            :start-after: # example-begin


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

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/reverse_input_iterator.py
            :language: python
            :start-after: # example-begin


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
        The code snippet below demonstrates the usage of a ``ReverseOutputIterator``:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/reverse_output_iterator.py
            :language: python
            :start-after: # example-begin


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
        by applying a transform operation before reducing the output:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/transform_iterator_basic.py
            :language: python
            :start-after: # example-begin


    Args:
        it: The iterator object to be transformed
        op: The transform operation

    Returns:
        A ``TransformIterator`` object to transform the items in ``it`` using ``op``
    """
    return make_transform_iterator(it, op)


def TransformOutputIterator(it, op):
    """Returns an Iterator representing a transformed sequence of output values.

    Similar to https://nvidia.github.io/cccl/thrust/api/classthrust_1_1transform__output__iterator.html

    Example:
        The code snippet below demonstrates the usage of ``TransformOutputIterator``.
        Before the result of the reduction is written to the output iterator,
        it is transformed by the function ``op``. Thus, the reduction operation
        computes the square root of the sum of of the input values.

    .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/transform_output_iterator.py
        :language: python

    Args:
        it: The iterator object to be transformed
        op: The transform operation

    Returns:
        A ``TransformOutputIterator`` object to transform the items in ``it`` using ``op``
    """
    return make_transform_iterator(it, op, IteratorIOKind.OUTPUT)


def ZipIterator(*iterators):
    """Returns an Iterator representing a zipped sequence of values from N iterators.

    Similar to https://nvidia.github.io/cccl/thrust/api/classthrust_1_1zip__iterator.html

    The resulting iterator yields gpu_struct objects with fields corresponding to each input iterator.
    For 2 iterators, fields are named 'first' and 'second'. For N iterators, fields are indexed
    as field_0, field_1, ..., field_N-1.

    Example:
        The code snippet below demonstrates the usage of a ``ZipIterator``
        combining two device arrays:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/zip_iterator_elementwise.py
            :language: python
            :start-after: # example-begin

    Args:
        *iterators: Variable number of iterators to zip (at least 1)

    Returns:
        A ``ZipIterator`` object that yields combined values from all input iterators
    """
    return make_zip_iterator(*iterators)
