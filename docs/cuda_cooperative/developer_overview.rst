Developer Overview
==================

This is the developer overview for the ``cuda.cooperative`` library.

How It Works
------------

Concepts
--------

Usage
^^^^^

Using the primitives from Numba CUDA kernels written in Python follows
the same general pattern:

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


Algorithms
----------

Loading & Storing
^^^^^^^^^^^^^^^^^

Sorting
^^^^^^^

Prefix Scans
^^^^^^^^^^^^

Reduction
^^^^^^^^^
.. vim: set filetype=rst expandtab ts=8 sw=2 sts=2 tw=72 ai:
