.. _cccl-python-compute:

``cuda.compute``: Parallel Computing Primitives
===============================================

The ``cuda.compute`` library provides parallel computing primitives that operate
on entire arrays or ranges of data. These algorithms are designed to be easy to use from Python
while delivering the performance of hand-optimized CUDA kernels, portable across different
GPU architectures.

Algorithms
----------

The core functionality provided by the ``cuda.compute`` library are algorithms such
as reductions, scans, sorts, and transforms.

Here's a simple example showing how to use the :func:`reduce_into <cuda.compute.algorithms.reduce_into>` algorithm to
reduce an array of integers.


.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/reduction/sum_reduction.py
   :language: python
   :start-after: # example-begin
   :caption: Basic reduction example.

Many algorithms, including reduction, require a temporary memory buffer.
The library will allocate this buffer for you, but you can also use the
object-based API for greater control.

.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/reduction/reduce_object.py
   :language: python
   :start-after: # example-begin
   :caption: Reduction with object-based API.


Iterators
---------

Algorithms can be used not just on arrays, but also on iterators. Iterators
provide a way to represent sequences of data without needing to allocate memory
for them.

Here's an example showing how to use reduction with a :func:`CountingIterator <cuda.compute.iterators.CountingIterator>` that
generates a sequence of numbers starting from a specified value.

.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/counting_iterator_basic.py
   :language: python
   :start-after: # example-begin
   :caption: Counting iterator example.

Iterators also provide a way to compose operations. Here's an example showing
how to use :func:`reduce_into <cuda.compute.algorithms.reduce_into>` with a :func:`TransformIterator <cuda.compute.iterators.TransformIterator>` to compute the sum of squares
of a sequence of numbers.

.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/transform_iterator_basic.py
   :language: python
   :start-after: # example-begin
   :caption: Transform iterator example.

Iterators that wrap an array (or another output iterator) may be used as both input and output iterators.
Here's an example showing how to use a
:func:`TransformIterator <cuda.compute.iterators.TransformIterator>` to transform the output
of a reduction before writing to the underlying array.

.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/transform_output_iterator.py
   :language: python
   :start-after: # example-begin
   :caption: Transform output iterator example.

Custom Types
------------

The ``cuda.compute`` library supports defining custom data types,
using the :func:`gpu_struct <cuda.compute.struct.gpu_struct>` decorator.
Here are some examples showing how to define and use custom types:

.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/struct/struct_reduction.py
   :language: python
   :start-after: # example-begin
   :caption: Custom type reduction example.

User-defined operations
-----------------------

A powerful feature of ``cuda.compute`` is the ability to customized algorithms
with user-defined operations. Below is an example of doing a custom reduction
with a user-defined binary operation.

.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/reduction/sum_custom_reduction.py
   :language: python
   :start-after: # example-begin
   :caption: Reduction with user-defined binary operations.

Note that user-defined operations are compiled into device code
using [``numba-cuda``](https://nvidia.github.io/numba-cuda/),
so many of the same features and restrictions of `numba` and `numba-cuda` apply.
Here are some important gotchas to be aware of:

* Lambda functions are not supported.
* Functions may invoke other functions, but the inner functions must be
  decorated with ``@numba.cuda.jit``.
* Functions capturing by global reference may not work as intended.
  Prefer using closures in these situations.

  Here is an example of a function that captures a global variable by reference,
  which is then used in a loop with ``unary_transform``.

  .. code-block:: python

     for i in range(10):
         def func(x):
             return x + i  # i is captured from global scope

         cuda.compute.unary_transform(d_in, d_out, func, num_items)

  Modifications to the global variable ``i`` may not be reflected in the function
  when the function is called multiple times. Thus, the different calls to
  ``unary_transform`` may not produce different results. This is true even though
  the function is redefined each time in the loop.

  To avoid this, capture the variable in a closure:

  .. code-block:: python

     def make_func(i):
         def func(x):
            return x + i  # i is captured as a closure variable
         return func

     for i in range(10):
         func = make_func(i)
         cuda.compute.unary_transform(d_in, d_out, func, num_items)


Example Collections
-------------------

For complete runnable examples and more advanced usage patterns, see our
full collection of `examples <https://github.com/NVIDIA/CCCL/tree/main/python/cuda_cccl/tests/compute/examples>`_.

External API References
------------------------

- :ref:`cuda_compute-module`
