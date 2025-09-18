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

.. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/reduction/sum_reduction.py
   :language: python
   :start-after: # example-begin
   :caption: Basic reduction example.

Many algorithms, including reduction, require a temporary memory buffer.
The library will allocate this buffer for you, but you can also use the
object-based API for greater control.

.. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/reduction/reduce_object.py
   :language: python
   :start-after: # example-begin
   :caption: Reduction with object-based API.


Iterators
---------

Algorithms can be used not just on arrays, but also on iterators. Iterators
provide a way to represent sequences of data without needing to allocate memory
for them.

Here's an example showing how to use reduction with a :func:`CountingIterator <cuda.cccl.parallel.experimental.iterators.CountingIterator>` that
generates a sequence of numbers starting from a specified value.

.. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/counting_iterator_basic.py
   :language: python
   :start-after: # example-begin
   :caption: Counting iterator example.

Iterators also provide a way to compose operations. Here's an example showing
how to use :func:`reduce_into <cuda.cccl.parallel.experimental.algorithms.reduce_into>` with a :func:`TransformIterator <cuda.cccl.parallel.experimental.iterators.TransformIterator>` to compute the sum of squares
of a sequence of numbers.

.. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/transform_iterator_basic.py
   :language: python
   :start-after: # example-begin
   :caption: Transform iterator example.

Iterators that wrap an array (or another output iterator) may be used as both input and output iterators.
Here's an example showing how to use a
:func:`TransformIterator <cuda.cccl.parallel.experimental.iterators.TransformIterator>` to transform the output
of a reduction before writing to the underlying array.

.. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/iterator/transform_output_iterator.py
   :language: python
   :start-after: # example-begin
   :caption: Transform output iterator example.

Custom Types
------------

The ``parallel`` library supports defining custom data types,
using the :func:`gpu_struct <cuda.cccl.parallel.experimental.struct.gpu_struct>` decorator.
Here are some examples showing how to define and use custom types:

.. literalinclude:: ../../python/cuda_cccl/tests/parallel/examples/reduction/pixel_reduction.py
   :language: python
   :start-after: # example-begin
   :caption: Custom type reduction example.


Example Collections
-------------------

For complete runnable examples and more advanced usage patterns, see our
full collection of `examples <https://github.com/NVIDIA/CCCL/tree/main/python/cuda_cccl/tests/parallel/examples>`_.

External API References
------------------------

- :ref:`cuda_parallel-module`
