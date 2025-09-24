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
        device_array: Array storing the input sequence of data items
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


def ReverseIterator(sequence):
    """Returns an Iterator over an array or another iterator in reverse.

    Similar to [std::reverse_iterator](https://en.cppreference.com/w/cpp/iterator/reverse_iterator).

    Examples:
        The code snippet below demonstrates the usage of a ``ReverseIterator`` as an input iterator:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/reverse_input_iterator.py
            :language: python
            :start-after: # example-begin

        The code snippet below demonstrates the usage of a ``ReverseIterator`` as an output iterator:

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/reverse_output_iterator.py
            :language: python
            :start-after: # example-begin


    Args:
        sequence: The iterator or array to be reversed

    Returns:
        A ``ReverseIterator`` object
    """
    return make_reverse_iterator(sequence)


def TransformIterator(it, op):
    """An iterator that applies a user-defined unary function to the elements of an underlying iterator as they are read.

    Similar to [thrust::transform_iterator](https://nvidia.github.io/cccl/thrust/api/classthrust_1_1transform__iterator.html)

    Example:
        The code snippet below demonstrates the usage of a ``TransformIterator`` composed with a ``CountingIterator``
        to transform the input before performing a reduction.

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/transform_iterator_basic.py
            :language: python
            :start-after: # example-begin
    Args:
        it: The underlying iterator
        op: The unary operation to be applied to values as they are read from ``it``

    Returns:
        A ``TransformIterator`` object to transform the items in ``it`` using ``op``
    """
    return make_transform_iterator(it, op, "input")


def TransformOutputIterator(it, op):
    """An iterator that applies a user-defined unary function to values before writing them to an underlying iterator.

    Similar to [thrust::transform_output_iterator](https://nvidia.github.io/cccl/thrust/api/classthrust_1_1transform__output__iterator.html).

    Example:
        The code snippet below demonstrates the usage of a ``TransformOutputIterator`` to transform the output
        of a reduction before writing to an output array.

        .. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/transform_output_iterator.py
            :language: python
            :start-after: # example-begin

    Args:
        it: The underlying iterator
        op: The operation to be applied to values before they are written to ``it``

    Returns:
        A ``TransformOutputIterator`` object that applies ``op`` to transform values before writing them to ``it``
    """
    return make_transform_iterator(it, op, "output")


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
