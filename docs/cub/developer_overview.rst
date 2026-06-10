CUB Developer Overview
##########################

.. toctree::
   :hidden:
   :maxdepth: 2

   developer/thread_level
   developer/warp_level
   developer/block_scope
   developer/device_scope
   developer/nvtx

This living document serves as a guide to the design of the internal structure of CUB.

CUB provides layered algorithms that correspond to the thread/warp/block/device hierarchy of threads in CUDA.
There are distinct algorithms for each layer and higher-level layers build on top of those below.

For example, CUB has four flavors of ``reduce``,
one for each layer: ``ThreadReduce, WarpReduce, BlockReduce``, and ``DeviceReduce``.
Each is unique in how it is invoked,
how many threads participate,
and on which thread(s) the result is valid.

These layers naturally build on each other.
For example, :cpp:struct:`cub::WarpReduce` uses :cpp:func:`cub::ThreadReduce`,
:cpp:struct:`cub::BlockReduce` uses :cpp:struct:`cub::WarpReduce`, etc.

:cpp:func:`cub::ThreadReduce`

   - A normal function invoked and executed sequentially by a single thread that returns a valid result on that thread
   - Single thread functions are usually an implementation detail and not exposed in CUB's public API

:cpp:struct:`cub::WarpReduce` and :cpp:struct:`cub::BlockReduce`

   - A "cooperative" function where threads concurrently invoke the same function to execute parallel work
   - The function's return value is well-defined only on the "first" thread (lowest thread index)

:cpp:struct:`cub::DeviceReduce`

   - A normal function invoked by a single thread that spawns additional threads to execute parallel work
   - Result is stored in the pointer provided to the function
   - Function returns a ``cudaError_t`` error code
   - Function does not synchronize the host with the device


The table below provides a summary of these functions:

.. list-table::
    :class: table-no-stripes
    :header-rows: 1

    * - layer
      - coop invocation
      - parallel execution
      - max threads
      - valid result in
    * - :cpp:func:`cub::ThreadReduce`
      - :math:`-`
      - :math:`-`
      - :math:`1`
      - invoking thread
    * - :cpp:struct:`cub::WarpReduce`
      - :math:`+`
      - :math:`+`
      - :math:`32`
      - main thread
    * - :cpp:struct:`cub::BlockReduce`
      - :math:`+`
      - :math:`+`
      - :math:`1024`
      - main thread
    * - :cpp:struct:`cub::DeviceReduce`
      - :math:`-`
      - :math:`+`
      - :math:`\infty`
      - global memory

The details of how each of these layers are implemented is described below.

Common Patterns
************************************

While CUB's algorithms are unique at each layer,
there are commonalities among all of them:

    - Algorithm interfaces are provided as *types* (classes)\ [1]_
    - Algorithms need temporary storage
    - Algorithms dispatch to specialized implementations depending on compile-time and runtime information
    - Cooperative algorithms require the number of threads at compile time (template parameter)

Invoking any CUB algorithm follows the same general pattern:

    #. Select the class for the desired algorithm
    #. Query the temporary storage requirements
    #. Allocate the temporary storage
    #. Pass the temporary storage to the algorithm
    #. Invoke it via the appropriate member function

An example of :cpp:struct:`cub::BlockReduce` demonstrates these patterns in practice:

.. code-block:: c++

    __global__ void kernel(int* per_block_results)
    {
      // (1) Select the desired class
      // `cub::BlockReduce` is a class template that must be instantiated for the
      // input data type and the number of threads. Internally the class is
      // specialized depending on the data type, number of threads, and hardware
      // architecture. Type aliases are often used for convenience:
      using BlockReduce = cub::BlockReduce<int, 128>;
      // (2) Query the temporary storage
      // The type and amount of temporary storage depends on the selected instantiation
      using TempStorage = typename BlockReduce::TempStorage;
      // (3) Allocate the temporary storage
      __shared__ TempStorage temp_storage;
      // (4) Pass the temporary storage
      // Temporary storage is passed to the constructor of the `BlockReduce` class
      BlockReduce block_reduce{temp_storage};
      // (5) Invoke the algorithm
      // The `Sum()` member function performs the sum reduction of `thread_data` across all 128 threads
      int thread_data[4] = {1, 2, 3, 4};
      int block_result = block_reduce.Sum(thread_data);

      per_block_results[blockIdx.x] = block_result;
    }

.. [1] Algorithm interfaces are provided as classes because it provides encapsulation for things like temporary storage requirements and enables partial template specialization for customizing an algorithm for specific data types or number of threads.

For more detailed descriptions of the respective algorithms levels see the individual sections below

  - :ref:`thread-level algorithms<cub-developer-guide-thread-level>`
  - :ref:`warp-level algorithms<cub-developer-guide-warp-level>`
  - :ref:`block-scope algorithms<cub-developer-guide-block-scope>`
  - :ref:`device-scope algorithms<cub-developer-guide-device-scope>`

There is additional information for :ref:`nvtx ranges <cub-developer-guide-nvtx>`
