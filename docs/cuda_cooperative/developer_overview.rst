Developer Overview
==================

This is the developer overview for the ``cuda.cooperative`` library.

How It Works
------------

Concepts
--------

Usage
^^^^^

Using the ``cuda.cooperative`` primitives from Numba CUDA kernels
written in Python follows the same general pattern:

  #. Identify the primitive(s) you need (e.g. load & store, reduction).
  #. `Create` a callable object `outside` of your Python CUDA kernel
     for all primitives you wish to use within your kernel.  This is
     done by invoking the corresponding Python function for the desired
     primitive, typically furnishing information about the data type and
     shape at a minimum, e.g.:

    .. code-block:: python

       from numba import cuda
       from cuda.cooperative.experimental import block

       # Create a callable for the block-wide load primitive.
       block_load = block.load(dtype=np.int32, threads_per_block=128)

       # Create a callable for the block-wide scan primitive.
       block_scan = block.scan(dtype=np.int32, threads_per_block=128)

       # These callables have important attributes that Numba CUDA
       # depends upon, such as ``.files`` and ``.temp_storage_bytes``.
       # They can't be called directly unless you're within a CUDA
       # kernel (i.e. a Python function decorated with a ``@cuda.jit``
       # decorator).

  #. `Decorate` your Python CUDA kernel with a
     ``@numba.cuda.jit(files=...)`` decorator, where the ``files``
     keyword argument reflects the set of all ``.files`` attributes
     on each of the returned callables, e.g.:

    .. code-block:: python

       @cuda.jit(files=block_load.files + block_scan.files)
       def kernel(d_in, d_out):
           ...

  #. Within your Python CUDA kernels, optionally create temporary
     storage by way of ``cuda.shared.array()`` or ``cuda.local.array()``
     helpers, e.g., from within the kernel:

    .. code-block:: python

       # Identify the largest temporary storage size required by all
       # primitives.
       max_temp_storage_bytes = max(block_load.temp_storage_bytes,
                                     block_scan.temp_storage_bytes)

       # Create temporary storage in shared memory.
       temp_storage = cuda.shared.array(
           shape=max_temp_storage_bytes,
           dtype=np.uint8,
       )


  #. Invoke the `callables` you obtained for the desired primitives
     as necessary within your kernel implementation, e.g.:

    .. code-block:: python

       # Load data from global memory into shared memory.
       block_load(d_in, temp_storage)

       # Perform a block-wide scan on the loaded data.
       block_scan(temp_storage, d_out)

  #. After you've defined your kernel, invoke it using Numba CUDA's
     support for launching CUDA kernels.

    .. code-block:: python

       # Launch the kernel with a grid of 1 block and 128 threads.
       kernel[1, 128](d_in, d_out)

Temporary Storage
^^^^^^^^^^^^^^^^^

Frequently, cooperative primitives need some form of temporary storage,
or scratch space, to perform their work.  This temporary storage is
ideally allocated in shared memory, but can also be allocated in local
(global) memory if necessary.

Shared memory, although significantly faster than global memory, is a
limited resource, and the amount of shared memory available to each
thread block is governed by multiple facets, such as GPU architecture
and kernel launch configuration.

Link-Time Optimization: Intermediate Representation (LTO-IR)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Link-time optimization (LTO) is a compiler optimization technique that
allows the compiler to analyze and optimize the entire program at once,
rather than optimizing each translation unit (source file) separately.

LTO-IR is a specific form of LTO that operates on the intermediate
representation (IR) of the code, and allows for Python Numba CUDA
kernels to run at speed-of-light performance.

Primitives
----------

The following sections describe the primitives exposed by
``cuda.cooperative`` from the perspective of the underlying intent of
that primitive, such as loading and storing data, reduction, prefix scan,
etc.

Each algorithm will typically have both a block-wide and warp-wide
implementation, and calling conventions are typically identical between
the two.  That is, examples demonstrating, for example, a block-wide
load of data, would work equally well for a warp-wide data load.

Loading & Storing
^^^^^^^^^^^^^^^^^
Overview
++++++++
Loading and storing data optimally is a crucial first step in writing
performant CUDA kernels.  The block-wide and warp-wide load and store
primitives ensure data movement is occurring in the most optimal fashion
for the underlying GPU architecture.  Specifically, they ensure reads
and writes are coalesced properly, such that the underlying bandwidth
backing either the shared memory or HBM global memory can be saturated.

Sorting
^^^^^^^
Overview
++++++++
Two sorting algorithms are provided: merge sort and radix sort.  The
calling conventions are practically identical between the two, only
differing in the type of algorithm that can be used.

Merge Sort
~~~~~~~~~~
Merge sort uses a parallel merging strategy, combining two sorted
sequences into a single sorted sequence.  It is a comparative sorting
algorithm, and thus, relies on the ability, given two elements, to
determine if one is greater than, equal, or less than the other.  This
means it can be used to sort custom types---provided that custom
comparators are furnished to the algorithm.  As a side-effect of the
comparative nature of the algorithm, merge sort can handle data types of
variable lengths

Merge sort is inherently stable, which means equal keys retain their
original order.

Merge sort is efficient when merging already-sorted data chunks (common
in some divide-and-conquer algorithms).  It offers also offers
predictable performance for partially-sorted or nearly-sorted data.

Merge sort performance will degrade depending on how poorly the data is
already sorted, with truly random data requiring the most amount of
computation to successfully sort.

Radix Sort
~~~~~~~~~~
Radix sort is a non-comparative sorting algorithm based on individual
digits, or bits, from least-significant to most-significant (or vice
versa).  It uses digit-wise bucketing and histogram computation in
parallel to efficiently sort numeric or key-value pairs.

Radix sort is optimal for fixed-size numeric data types.  It is usually
faster on uniformly random or large datasets due to requiring fewer
passes and simpler computation logic.

Radix sort offers no stability guarantees unless explicitly requested.

Radix sort performance can degrade for non-uniform data distributions or
many unique values, as overhead is introduced by uneven binning.

Choosing Between Merge vs Radix Sort
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Comparison of Block-wide and Warp-wide Sorting in CUB
   :header-rows: 1
   :widths: 25 35 35

   * - Feature
     - Merge Sort
     - Radix Sort

   * - **Stability**
     - Naturally stable
     - Typically stable if implemented properly

   * - **Key Types**
     - Arbitrary comparable types
     - Primarily numeric types

   * - **Data Characteristics**
     - Efficient on partially or nearly sorted data
     - Efficient on random, uniformly distributed numeric data

   * - **Memory Usage**
     - Higher due to merging buffers
     - Moderate; depends on histogram/bin size

   * - **Performance Predictability**
     - Predictable for partially sorted data
     - Predictable and typically faster for numeric data

   * - **Implementation Complexity**
     - Moderate complexity
     - Slightly higher complexity, especially for custom types

Examples
++++++++

References
++++++++++

Python API References:

C++ API References:

Parallel Prefix Scans
^^^^^^^^^^^^^^^^^^^^^
Parallel prefix scans compute cumulative operations across elements of
an array.  The most well-known parallel prefix scan primitives are the
inclusive and exclusive sum: when provided an array of numerical data,
each element of the returned array will reflect the sum of itself and
all prior elements.

Reduction
^^^^^^^^^

Exchange
^^^^^^^^

Adjacent Differences
^^^^^^^^^^^^^^^^^^^^

.. vim: set filetype=rst expandtab ts=8 sw=2 sts=2 tw=72 ai:
