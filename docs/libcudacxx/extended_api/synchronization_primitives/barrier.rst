.. _libcudacxx-extended-api-synchronization-barrier:

``cuda::barrier``
=================

.. toctree::
   :hidden:
   :maxdepth: 1

   barrier/init
   barrier/barrier_native_handle
   barrier/barrier_arrive_tx
   barrier/barrier_expect_tx

Defined in header ``<cuda/barrier>``:

.. code:: cuda

   template <cuda::thread_scope Scope,
             typename CompletionFunction = /* unspecified */>
   class cuda::barrier;

The class template ``cuda::barrier`` is an extended form of `cuda::std::barrier <https://en.cppreference.com/w/cpp/thread/barrier>`_
that takes an additional :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` argument.

If ``!(scope == cuda::thread_block_scope && cuda::device::is_address_from(this, cuda::device::address_space::shared))``, then the semantics are the same as
`cuda::std::barrier <https://en.cppreference.com/w/cpp/thread/barrier>`_, otherwise, see below.

The ``cuda::barrier`` class templates extends ``cuda::std::barrier`` with the following additional operations:

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - :ref:`cuda::barrier::init <libcudacxx-extended-api-synchronization-barrier-barrier-init>`
     - Initialize a ``cuda::barrier``.
   * - :ref:`cuda::device::barrier_native_handle <libcudacxx-extended-api-synchronization-barrier-barrier-native-handle>`
     - Get the native handle to a ``cuda::barrier``.
   * - :ref:`cuda::device::barrier_arrive_tx <libcudacxx-extended-api-synchronization-barrier-barrier-arrive-tx>`
     - Arrive on a ``cuda::barrier<cuda::thread_scope_block>`` with transaction count update.
   * - :ref:`cuda::device::barrier_expect_tx <libcudacxx-extended-api-synchronization-barrier-barrier-expect-tx>`
     - Update transaction count of ``cuda::barrier<cuda::thread_scope_block>``.

If ``scope == cuda::thread_scope_block && cuda::device::is_address_from(this, cuda::device::address_space::shared)``, then the semantics of `[thread.barrier.class] <http://eel.is/c++draft/thread.barrier.class>`_
of ISO/IEC IS 14882 (the C++ Standard) are modified as follows:

   A barrier is a thread coordination mechanism whose lifetime consists
   of a sequence of barrier phases, where each phase allows at most an
   expected number of threads to block until the expected number of
   threads **and the expected number of transaction-based asynchronous
   operations** arrive at the barrier. Each *barrier phase* consists of
   the following steps:

   1. The *expected count* is decremented by each call to ``arrive``,\ ``arrive_and_drop``\ **,
      or cuda::device::barrier_arrive_tx**.
   2. **The transaction count is incremented by each call to cuda::device::barrier_arrive_tx and decremented by the
      completion of transaction-based asynchronous operations such as cuda::memcpy_async_tx.**
   3. Exactly once after **both** the *expected count* **and the transaction count** reach zero, a thread executes the
      *completion step* during its call to ``arrive``, ``arrive_and_drop``, ``cuda::device::barrier_arrive_tx``,
      or ``wait``, except that it is implementation-defined whether the step executes if no thread calls ``wait``.
   4. When the completion step finishes, the *expected count* is reset to what was specified by the ``expected``
      argument to the constructor, possibly adjusted by calls to ``arrive_and_drop``, and the next phase starts.

   Concurrent invocations of the member functions of barrier **and the non-member barrier APIs in cuda::device**,
   other than its destructor, do not introduce data races. The member functions ``arrive`` and ``arrive_and_drop``,
   **and the non-member function cuda::device::barrier_arrive_tx**, execute atomically.

.. rubric:: NVCC ``__shared__`` Initialization Warnings

When using libcu++ with NVCC, a ``__shared__`` ``cuda::barrier`` will lead to the following warning because
``__shared__`` variables are not initialized:

.. code:: bash

   warning: dynamic initialization is not supported for a function-scope static
   __shared__ variable within a __device__/__global__ function

It can be silenced using ``#pragma nv_diag_suppress static_var_with_dynamic_init``.

To properly initialize a ``__shared__`` ``cuda::barrier``, use the
:ref:`cuda::barrier::init <libcudacxx-extended-api-synchronization-barrier-barrier-init>` friend function.

.. rubric:: Concurrency Restrictions

An object of type ``cuda::barrier`` or ``cuda::std::barrier`` shall not be accessed concurrently by CPU and GPU threads unless:

   - it is in unified memory and the `concurrentManagedAccess property <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b>`_
     is 1, or
   - it is in CPU memory and the `hostNativeAtomicSupported property <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f>`_
     is 1.

Note, for objects of scopes other than ``cuda::thread_scope_system`` this is a data-race, and therefore also prohibited
regardless of memory characteristics.

Under CUDA Compute Capability 8 (Ampere) or above, when an object of type ``cuda::barrier<thread_scope_block>`` is
placed in ``__shared__`` memory, the member function ``arrive`` performs a reduction of the arrival count among
`coalesced threads <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#coalesced-group-cg>`_ followed
by the arrival operation in one thread. Programs shall ensure that this transformation would not introduce errors,
for example relative to the requirements of `thread.barrier.class paragraph 12 <https://eel.is/c++draft/thread.barrier.class#12>`_
of ISO/IEC IS 14882 (the C++ Standard).

Under CUDA Compute Capability 6 (Pascal) or prior, an object of type ``cuda::barrier`` or ``cuda::std::barrier`` may
not be used.

.. rubric:: Shared memory barriers with transaction count

In addition to the arrival count, a ``cuda::barrier<thread_scope_block>`` object located in shared memory supports a
`tx-count <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-tracking-async-operations>`_,
which is used for tracking the completion of some asynchronous memory operations or transactions.
The tx-count tracks the number of asynchronous transactions, in units specified by the asynchronous memory operation
(typically bytes), that are outstanding and yet to be complete. This capability is exposed, starting with the Hopper
architecture (CUDA Compute Capability 9).

The tx-count of ``cuda::barrier`` must be set to the total amount of asynchronous memory operations, in units as
specified by the asynchronous operations, to be tracked by the current phase. This can be achieved with the
:ref:`cuda::device::barrier_arrive_tx <libcudacxx-extended-api-synchronization-barrier-barrier-arrive-tx>` function call.

Upon completion of each of the asynchronous operations, the tx-count of the ``cuda::barrier`` will be updated and thus
progress the ``cuda::barrier`` towards the completion of the current phase. This may complete the current phase.

.. rubric:: Implementation-Defined Behavior

For each :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` ``S`` and ``CompletionFunction``
``F``, the value of ``cuda::barrier<S, F>::max()`` is as follows:

.. list-table::
   :widths: 25 25 50
   :header-rows: 0

   * - :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` ``S``
     - ``CompletionFunction`` ``F``
     - ``barrier<S, F>::max()``
   * - ``cuda::thread_scope_block``
     - Default or user-provided
     - ``(1 << 20) - 1``
   * - *Not* ``cuda::thread_scope_block``
     - Default
     - ``cuda::std::numeric_limits<cuda::std::int32_t>::max()``
   * - *Not* ``cuda::thread_scope_block``
     - User-provided
     - ``cuda::std::numeric_limits<cuda::std::ptrdiff_t>::max()``

.. rubric:: Example

.. code:: cuda

   #include <cuda/barrier>

   __global__ void example_kernel() {
     // This barrier is suitable for all threads in the system.
     cuda::barrier<cuda::thread_scope_system> a(10);

     // This barrier has the same type as the previous one (`a`).
     cuda::std::barrier<> b(10);

     // This barrier is suitable for all threads on the current processor (e.g. GPU).
     cuda::barrier<cuda::thread_scope_device> c(10);

     // This barrier is suitable for all threads in the same thread block.
     cuda::barrier<cuda::thread_scope_block> d(10);
   }

`See it on Godbolt <https://godbolt.org/z/7Kbz5qqhh>`_

.. rubric:: Example: 1D TMA load of two buffers with arrival token (sm90+)

The following example shows how to use TMA to load two tiles of data from global memory into shared memory:

.. code:: cuda

   #include <cuda/barrier>
   #include <cuda/ptx>

   // selects a single leader thread from the block
   __device__ bool elect_one() {
     const unsigned int tid = threadIdx.x;
     const unsigned int warp_id = tid / 32;
     const unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // broadcast from lane 0
     return uniform_warp_id == 0 && cuda::ptx::elect_sync(0xFFFFFFFF); // elect a leader thread among warp 0
   }

   __global__ void example_kernel(int* gmem1, double* gmem2) {
     constexpr int tile_size = 1024;
     __shared__ alignas(16)    int smem1[tile_size];
     __shared__ alignas(16) double smem2[tile_size];

     #pragma nv_diag_suppress static_var_with_dynamic_init
     __shared__  cuda::barrier<cuda::thread_scope_block> bar;

     // setup the barrier where each thread in the block arrives at
     if (threadIdx.x == 0) {
       init(&bar, blockDim.x);
     }
     __syncthreads(); // need to sync so other threads can arrive

     // select a single thread from the block and issue two TMA bulk copy operations
     const auto elected = elect_one();
     if (elected) {
       cuda::device::memcpy_async_tx(smem1, gmem1, cuda::aligned_size_t<16>(tile_size * sizeof(int)   ), bar);
       cuda::device::memcpy_async_tx(smem2, gmem2, cuda::aligned_size_t<16>(tile_size * sizeof(double)), bar);
     }

     // arrive at the barrier
     // the elected thread also updates the barrier's expect_tx with the **total** number of loaded bytes
     const int tx_count = elected ? tile_size * (sizeof(int) + sizeof(double)) : 0;
     auto token = cuda::device::barrier_arrive_tx(bar, 1, tx_count);

     // wait for TMA copies to complete
     bar.wait(cuda::std::move(token));

     // process data in smem ...
   }

`See it on Godbolt <https://godbolt.org/z/ddhzGeWPE>`_

Data is loaded from the global memory pointers ``gmem1`` and ``gmem2``
into the shared memory buffers ``smem1`` and ``smem2``.
The shared memory buffers have to be aligned to 16 bytes.
Additionally, a single barrier with block scope is setup by a single leader thread.
Each thread will arrive at the barrier, so ``blockDim.x`` is passed as arrival count to ``init``.
This barrier is used to synchronize the asynchronous TMA copies with the rest of the threads in the block.
The leader initiates the TMA copies using ``ptx::cp_async_bulk``,
and updates the barrier's tx-count with the total number of bytes transferred by the TMA copy operations
while arriving at the barrier using ``cuda::device::barrier_arrive_tx``.
All other threads just arrive normally at the barrier.
All threads then wait on the barrier to complete the current phase by calling ``wait``.
Once ``wait`` returns, all data is available in shared memory.

.. rubric:: Example: 1D TMA load of two buffers with barrier phase parity check (sm90+)

.. code:: cuda

   #include <cuda/barrier>
   #include <cuda/ptx>

   // selects a single leader thread from the block
   __device__ bool elect_one() {
     const unsigned int tid = threadIdx.x;
     const unsigned int warp_id = tid / 32;
     const unsigned int uniform_warp_id = __shfl_sync(0xFFFFFFFF, warp_id, 0); // broadcast from lane 0
     return uniform_warp_id == 0 && cuda::ptx::elect_sync(0xFFFFFFFF); // elect a leader thread among warp 0
   }

   __global__ void example_kernel(int* gmem1, double* gmem2) {
     constexpr int tile_size = 1024;
     __shared__ alignas(16)    int smem1[tile_size];
     __shared__ alignas(16) double smem2[tile_size];

     #pragma nv_diag_suppress static_var_with_dynamic_init
     __shared__  cuda::barrier<cuda::thread_scope_block> bar;

     // setup the barrier where only the leader thread arrives
     if (elect_one()) {
       init(&bar, 1);

       // issue two TMA bulk copy operations
       cuda::device::memcpy_async_tx(smem1, gmem1, cuda::aligned_size_t<16>(tile_size * sizeof(int)   ), bar);
       cuda::device::memcpy_async_tx(smem2, gmem2, cuda::aligned_size_t<16>(tile_size * sizeof(double)), bar);

       // arrive and update the barrier's expect_tx with the **total** number of loaded bytes
       (void)cuda::device::barrier_arrive_tx(bar, 1, tile_size * (sizeof(int) + sizeof(double)));
     }
     __syncthreads(); // need to sync so the barrier is set up when the other threads arrive and wait

     // wait for the current barrier phase to complete
     bar.wait_parity(0);

     // process data in smem ...
   }

`See it on Godbolt <https://godbolt.org/z/oq85PoKKj>`_

This is similar to the above example, but this time only the leader thread arrives at the barrier.
This has the advantage that only one leader election is necessary
and a single uniform datapath section is generated.
This generally generates less instructions.
Because now we don't get an arrival token in each thread, we cannot use ``wait(token)`` with all threads.
Instead, we just wait until the end of the barrier's current phase using ``wait_parity(0)``
(0 is the parity of the current phase).

Before CCCL 3.2, ``bar.wait_parity(0);`` contained additional logic which may have lead to unnecessary instructions.
If you are using CCCL below 3.2, you may replace this line with:

.. code:: cuda

   while (!cuda::ptx::mbarrier_try_wait_parity(cuda::device::barrier_native_handle(bar), 0))
     ;

.. rubric:: Example: 1D TMA load and store using `cuda::device::memcpy_async_tx` (sm90+)

This example can be found in :ref:`libcudacxx-extended-api-asynchronous-operations-memcpy-async-tx-example`.


.. rubric:: Example: 1D TMA load and store using `cuda::memcpy_async` (sm90+)

This example can be found in the
`CUDA programming guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/#using-tma-to-transfer-one-dimensional-arrays>`_.
