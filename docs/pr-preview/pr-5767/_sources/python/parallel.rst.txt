.. _cccl-python-parallel:

``parallel``: Device-Level Parallel Algorithms
==============================================

The ``cuda.cccl.parallel`` library provides device-level algorithms that operate
on entire arrays or ranges of data. These algorithms are designed to be easy to use from Python
while delivering the performance of hand-optimized CUDA kernels, portable across different
GPU architectures.

Algorithms
----------

The core functionality provided by the ``parallel`` library are algorithms such
as reductions, scans, sorts, and transforms.

Here's a simple example showing how to use the :func:`reduce_into <cuda.cccl.parallel.experimental.algorithms.reduce_into>` algorithm to
reduce an array of integers.

.. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/reduction/basic_reduction.py
   :language: python
   :pyobject: sum_reduction_example
   :caption: Basic reduction example. `View complete source on GitHub <https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/parallel/examples/reduction/basic_reduction.py>`__

Iterators
---------

Algorithms can be used not just on arrays, but also on iterators. Iterators
provide a way to represent sequences of data without needing to allocate memory
for them.

Here's an example showing how to use reduction with a :func:`CountingIterator <cuda.cccl.parallel.experimental.iterators.CountingIterator>` that
generates a sequence of numbers starting from a specified value.

.. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/counting_iterator.py
   :language: python
   :pyobject: counting_iterator_example
   :caption: Counting iterator example. `View complete source on GitHub <https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/parallel/examples/iterator/counting_iterator.py>`__

Iterators also provide a way to compose operations. Here's an example showing
how to use :func:`reduce_into <cuda.cccl.parallel.experimental.algorithms.reduce_into>` with a :func:`TransformIterator <cuda.cccl.parallel.experimental.iterators.TransformIterator>` to compute the sum of squares
of a sequence of numbers.

.. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/transform_iterator.py
   :language: python
   :pyobject: transform_iterator_example
   :caption: Transform iterator example. `View complete source on GitHub <https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/parallel/examples/iterator/transform_iterator.py>`__

Custom Types
------------

The ``parallel`` library supports defining custom data types,
using the :func:`gpu_struct <cuda.cccl.parallel.experimental.struct.gpu_struct>` decorator.
Here are some examples showing how to define and use custom types:

.. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/reduction/custom_types.py
   :language: python
   :pyobject: pixel_reduction_example
   :caption: Custom type reduction example. `View complete source on GitHub <https://github.com/NVIDIA/cccl/blob/main/python/cuda_cccl/tests/parallel/examples/reduction/custom_types.py>`__


Example Collections
-------------------

For complete runnable examples and more advanced usage patterns, see our
full collection of `examples <https://github.com/NVIDIA/CCCL/tree/main/python/cuda_cccl/tests/parallel/examples>`_.

External API References
------------------------

- :ref:`cuda_parallel-module`
