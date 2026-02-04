.. _cccl-python-compute:

``cuda.compute``: Parallel Computing Primitives
===============================================

The ``cuda.compute`` library provides composable primitives for building custom
parallel algorithms on the GPU—without writing CUDA kernels directly.

Algorithms
----------

Algorithms are the core of ``cuda.compute``. They operate on arrays or
:ref:`iterators <cuda.compute.iterators>` and can be composed to build specialized
GPU operations—reductions, scans, sorts, transforms, and more.

Typical usage of an algorithm looks like this:

.. code-block:: python

   cuda.compute.reduce_into(
      d_in=...,       # input array or iterator
      d_out=...,      # output array or iterator
      op=...,         # binary operator (built-in or user-defined)
      num_items=...,  # number of input elements
      h_init=...,     # initial value for the reduction
   )

API conventions
+++++++++++++++

* **Naming** — The ``d_`` prefix denotes *device* memory (e.g., CuPy arrays, PyTorch tensors);
  ``h_`` denotes *host* memory (NumPy arrays). Some scalar values must be passed as
  host arrays.

* **Output semantics** — Algorithms write results into a user-provided array or iterator
  rather than returning them. This keeps memory ownership explicit and lifetimes under
  your control.

* **Operators** — Many algorithms accept an ``op`` parameter. This can be a built-in
  :class:`OpKind <cuda.compute.op.OpKind>` value or a
  :ref:`user-defined function <cuda.compute.user_defined_operations>`.
  When possible, prefer built-in operators (e.g., ``OpKind.PLUS``) over the equivalent
  user-defined operation (e.g., ``lambda a, b: a + b``) for better performance.

* **Iterators** — Inputs and outputs can be :ref:`iterators <cuda.compute.iterators>`
  instead of arrays, enabling lazy evaluation and operation fusion.

Full Example
++++++++++++

The following example uses :func:`reduce_into <cuda.compute.algorithms.reduce_into>`
to compute the sum of a sequence of integers:

.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/reduction/sum_reduction.py
   :language: python
   :start-after: # example-begin
   :caption: Sum reduction example.

Object-based API (expert mode)
++++++++++++++++++++++++++++++

Many algorithms allocate temporary device memory for intermediate results. For finer
control over allocation—or to reuse buffers across calls—use the object-based API.
For example, :func:`make_reduce_into <cuda.compute.algorithms.make_reduce_into>`
returns a reusable reduction object that lets you manage memory explicitly.

.. code-block:: python
   :caption: Controlling temporary memory.

   # create a reducer object:
   reducer = cuda.compute.make_reduce_into(d_in, d_out, op, h_init)
   # get the temporary storage size by passing None as the first argument:
   temp_storage_bytes = reducer(None, d_in, d_out, op, num_items, h_init)
   # allocate the temporary storage as any array-like object
   # (e.g., CuPy array, Torch tensor):
   temp_storage = cp.empty(temp_storage_bytes, dtype=np.uint8)
   # perform the reduction, passing the temporary storage as the first argument:
   reducer(temp_storage, d_in, d_out, op, num_items, h_init)

The object-based API splits the algorithm invocation into three phases,

1. Constructing an algorithm object
2. Determining the amount of temporary memory needed by the computation
3. Performing the computation

It is important that the type of arguments passed during construction (step 1) match
those passed during invocation (step 2 and 3). Otherwise you may see unexpected errors
or silent bugs.

- Data types of arrays/iterators must match. If you pass an array of `int32` data type
  as the `d_in=` argument during construction of a reducer object, you must pass
  an array of dtype `int32` when invoking it. The array can be of a different size.

- Bytecode instructions of functions must match. If you pass a function/lambda for
  the operator during construction, you must pass a function with the same bytecode
  instructions during invocation. This means you _can_ pass a different function
  referencing different global/closures, but the operations within the functions
  must be the same.


.. _cuda.compute.user_defined_operations:

User-Defined Operations
-----------------------

A powerful feature is the ability to use algorithms with user-defined operations.
For example, to compute the sum of only the even values in a sequence,
we can use :func:`reduce_into <cuda.compute.algorithms.reduce_into>` with a custom binary operation:

.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/reduction/sum_custom_reduction.py
   :language: python
   :start-after: # example-begin
   :caption: Reduction with a custom binary operation.

Features and Restrictions
+++++++++++++++++++++++++

User-defined operations are compiled into device code using
`Numba CUDA <https://nvidia.github.io/numba-cuda/>`_, so they inherit many
of the same features and restrictions as Numba CUDA functions:

* `Python features <https://nvidia.github.io/numba-cuda/user/cudapysupported.html>`_
  and `atomic operations <https://nvidia.github.io/numba-cuda/user/intrinsics.html>`_
  supported by Numba CUDA are also supported within user-defined operations.
* Nested functions must be decorated with ``@numba.cuda.jit``.
* Variables captured in closures or globals follow
  `Numba CUDA semantics <https://nvidia.github.io/numba-cuda/user/globals.html>`_:
  scalars and host arrays are captured by value (as constants),
  while device arrays are captured by reference.

.. _cuda.compute.iterators:

Iterators
---------

Iterators represent sequences whose elements are computed **on the fly**. They can
be used in place of arrays in most algorithms, enabling lazy evaluation, operation
fusion, and custom data access patterns.

A :func:`CountingIterator <cuda.compute.iterators.CountingIterator>`, for example,
represents an integer sequence starting from a given value:

.. code-block:: python

   it = CountingIterator(np.int32(1))  # represents [1, 2, 3, 4, ...]

To compute the sum of the first 100 integers, we can pass a
:func:`CountingIterator <cuda.compute.iterators.CountingIterator>` directly to
:func:`reduce_into <cuda.compute.algorithms.reduce_into>`. No memory is allocated
to store the input sequence—the values are generated as needed.

.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/counting_iterator_basic.py
   :language: python
   :start-after: # example-begin
   :caption: Counting iterator example.

Iterators can also be used to *fuse* operations. In the example below, a
:func:`TransformIterator <cuda.compute.iterators.TransformIterator>` lazily applies
the square operation to each element of the input sequence. The resulting iterator
is then passed to :func:`reduce_into <cuda.compute.algorithms.reduce_into>` to compute
the sum of squares.

Because the square is evaluated on demand during the reduction, there is no need
to create or store an intermediate array of squared values. The transform and the
reduction are fused into a single pass over the data.

.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/transform_iterator_basic.py
   :language: python
   :start-after: # example-begin
   :caption: Transform iterator example.

Some iterators can also be used as the output of an algorithm. In the example below,
a :func:`TransformOutputIterator <cuda.compute.iterators.TransformOutputIterator>`
applies the square-root operation to the result of a reduction before writing
it into the underlying array.

.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/transform_output_iterator.py
   :language: python
   :start-after: # example-begin
   :caption: Transform output iterator example.

As another example, :func:`ZipIterator <cuda.compute.iterators.ZipIterator>` combines multiple
arrays or iterators into a single logical sequence. In the example below, we combine
a counting iterator and an array, creating an iterator that yields ``(index, value)``
pairs. This combined iterator is then used as the input to
:func:`reduce_into <cuda.compute.algorithms.reduce_into>` to compute the index of
the maximum value in the array.

.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/zip_iterator_counting.py
   :language: python
   :start-after: # example-begin
   :caption: Argmax using a zip iterator.

These examples illustrate a few of the patterns enabled by iterators. See the
:ref:`API reference <cuda_compute-module>` for the full set of available iterators.

.. _cuda.compute.custom_types:

Struct Types
------------

The :func:`gpu_struct <cuda.compute.struct.gpu_struct>` decorator defines
GPU-compatible struct types. These are useful when you have data laid out
as an "array of structures", similar to `NumPy structured arrays <https://numpy.org/doc/stable/user/basics.rec.html>`_.

.. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/struct/struct_reduction.py
   :language: python
   :start-after: # example-begin
   :caption: Custom struct type in a reduction.

Array of Structures vs Structure of Arrays
++++++++++++++++++++++++++++++++++++++++++

When working with structured data, there are two common memory layouts:

* **Array of Structures (AoS)** — each element is a complete struct, stored
  contiguously. For example, an array of ``Point`` structs where each point's
  ``x`` and ``y`` are adjacent in memory.

* **Structure of Arrays (SoA)** — each field is stored in its own array.
  For example, separate ``x_coords`` and ``y_coords`` arrays.

``cuda.compute`` supports both layouts:

* **``gpu_struct``** — defines a true AoS type with named fields
* **``ZipIterator``** — combines separate arrays into tuples on the fly, letting
  you work with SoA data as if it were AoS

.. _cuda.compute.caching:

Caching
-------

Algorithms in ``cuda.compute`` are compiled to GPU code at runtime. To avoid
recompiling on every call, build results are cached in memory. When you invoke
an algorithm with the same configuration—same dtypes, iterator kinds, operator,
and compute capability—the cached build is reused.

What determines the cache key
+++++++++++++++++++++++++++++

Each algorithm computes a cache key from:

* **Array dtypes** — the data types of input and output arrays
* **Iterator kinds** — for iterator inputs/outputs, a descriptor of the iterator type
* **Operator identity** — for user-defined functions, the function's bytecode,
  constants, and closure contents (see below)
* **Compute capability** — the GPU architecture of the current device
* **Algorithm-specific parameters** — such as initial value dtype or determinism mode

Note that array *contents* or *pointers* are not part of the cache key—only
the array's dtype. This means you can reuse a cached algorithm across different
arrays of the same type.

How user-defined functions are cached
+++++++++++++++++++++++++++++++++++++

User-defined operators and predicates are hashed based on their bytecode, constants,
and closure contents. Two functions with identical bytecode and closures produce
the same cache key, even if defined at different source locations.

Closure contents are recursively hashed:

* **Scalars and host arrays** — hashed by value
* **Device arrays** — hashed by pointer, shape, and dtype (not contents)
* **Nested functions** — hashed by their own bytecode and closures

Because device arrays captured in closures are hashed by pointer, changing the
array's contents does not invalidate the cache—only reassigning the variable to
a different array does.

Memory considerations
+++++++++++++++++++++

The cache persists for the lifetime of the process and grows with the number of
unique algorithm configurations. In long-running applications or exploratory
notebooks, this can accumulate significant memory.

To clear all caches and free memory:

.. code-block:: python

   import cuda.compute
   cuda.compute.clear_all_caches()

This forces recompilation on the next algorithm invocation—useful for benchmarking
compilation time or reclaiming memory.

Examples
--------

For complete runnable examples and additional usage patterns, see the
`examples directory <https://github.com/NVIDIA/CCCL/tree/main/python/cuda_cccl/tests/compute/examples>`_.

API Reference
-------------

- :ref:`cuda_compute-module`
