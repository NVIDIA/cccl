.. _libcudacxx-extended-api-memory-model:

Memory model
============

Standard C++ presents a view that the cost to synchronize threads is uniform and low.

CUDA C++ is different: the cost to synchronize threads grows as threads are further apart. It is low across threads
within a block, but high across arbitrary threads in the system running on multiple GPUs and CPUs.

To account for non-uniform thread synchronization costs that are not always low, CUDA C++ extends the standard C++
memory model and concurrency facilities in the ``cuda::`` namespace with **thread scopes**, retaining the syntax and
semantics of standard C++ by default.

.. _libcudacxx-extended-api-memory-model-thread-scopes:

Thread Scopes
-------------

A **thread scope** specifies the kind of threads that can synchronize with each other using a synchronization primitive such
as :ref:`atomic <libcudacxx-extended-api-synchronization-atomic>` or
:ref:`barrier <libcudacxx-extended-api-synchronization-barrier>`.

.. code:: cuda

   namespace cuda {

   enum thread_scope {
     thread_scope_system,
     thread_scope_device,
     thread_scope_block,
     thread_scope_thread
   };

   }  // namespace cuda

Scope Relationships
~~~~~~~~~~~~~~~~~~~

Each program thread is related to each other program thread by one or more thread scope relations:
   - Each thread in the system is related to each other thread in the system by the *system* thread scope:
     ``thread_scope_system``.
   - Each GPU thread is related to each other GPU thread in the same CUDA device and within the same `memory
     synchronization domain <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-synchronization-domains>`__
     by the *device* thread scope: ``thread_scope_device``.
   - Each GPU thread is related to each other GPU thread in the same CUDA thread block by the *block* thread scope:
     ``thread_scope_block``.
   - Each thread is related to itself by the ``thread`` thread scope: ``thread_scope_thread``.

Synchronization primitives
--------------------------

Types in namespaces ``std::`` and ``cuda::std::`` have the same behavior as corresponding types in namespace ``cuda::``
when instantiated with a scope of ``cuda::thread_scope_system``.

Atomicity
---------

An atomic operation is atomic at the scope it specifies if:

   - it specifies a scope other than ``thread_scope_system``, **or**
   - the scope is ``thread_scope_system`` and:

      -  it affects an object in `system allocated memory <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd>`__ and `pageableMemoryAccess <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cddc80992427a92713e699953a6d249d6f>`__ is ``1`` [0],  **or**
      -  it affects an object in `managed
         memory <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd>`__
         and
         `concurrentManagedAccess <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_116f9619ccc85e93bc456b8c69c80e78b>`__
         is ``1``, **or**
      -  it affects an object in `mapped
         memory <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory>`__ and
         `hostNativeAtomicSupported <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f>`__
         is ``1``, **or**
      -  it is a load or store that affects a naturally-aligned object of
         sizes ``1``, ``2``, ``4``, ``8``, or ``16`` bytes on `mapped
         memory <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory>`__ [1],
         **or**
      -  it affects an object in GPU memory, only GPU threads access it, and
          - `*val` returned from `cudaDeviceGetP2PAttribute(&val, cudaDevP2PAttrNativeAtomicSupported, srcDev, dstDev) <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g2f597e2acceab33f60bd61c41fea0c1b>`__ between each accessing `srcDev` and the GPU where the object resides, `dstDev`, is ``1``, or
          - only GPU threads from a single GPU concurrently access it.

.. note::
   - [0] If `PageableMemoryAccessUsesHostPagetables <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gg49e2f8c2c0bd6fe264f2fc970912e5cdc228cf8983c97d0e035da72a71494eaa>`__ is ``0`` then atomic operations to memory mapped file or ``hugetlbfs`` allocations are not atomic.
   - [1] If `hostNativeAtomicSupported <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_1ef82fd7d1d0413c7d6f33287e5b6306f>`__ is ``0``, atomic load or store operations at system scope that affect a
     naturally-aligned 16-byte wide object in
     `unified memory <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd>`__ or
     `mapped memory <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory>`__ require system
     support. NVIDIA is not aware of any system that lacks this support and there is no CUDA API query available to
     detect such systems.

Refer to the `CUDA programming guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html>`__
for more information on
`system allocated memory <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd>`__,
`managed memory <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd>`__,
`mapped memory <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory>`__,
CPU memory, and GPU memory.

Data Races
----------

Modify `intro.races paragraph 21 <https://eel.is/c++draft/intro.races#21>`__ of ISO/IEC IS 14882 (the C++ Standard)
as follows:

   The execution of a program contains a data race if it contains two potentially concurrent conflicting actions, at
   least one of which is not atomic **at a scope that includes the thread that performed the other operation**, and neither
   happens before the other, except for the special case for signal handlers described below.
   Any such data race results in undefined behavior. […]

Modify `thread.barrier.class paragraph 4 <https://eel.is/c++draft/thread.barrier.class#4>`__ of ISO/IEC IS
14882 (the C++ Standard) as follows

   4. Concurrent invocations of the member functions of ``barrier``, other than its destructor, do not introduce data
   races **as if they were atomic operations**. […]

Modify `thread.latch.class paragraph 2 <https://eel.is/c++draft/thread.latch.class#2>`__ of ISO/IEC IS 14882
(the C++ Standard) as follows:

   2. Concurrent invocations of the member functions of ``latch``, other than its destructor, do not introduce data
   races **as if they were atomic operations**. […]

Modify `thread.sema.cnt paragraph 3 <https://eel.is/c++draft/thread.sema.cnt#3>`__ of ISO/IEC IS 14882
(the C++ Standard) as follows:

   3. Concurrent invocations of the member functions of ``counting_semaphore``, other than its destructor, do not
   introduce data races **as if they were atomic operations**.

Modify `thread.stoptoken.intro paragraph 5 <https://eel.is/c++draft/thread#stoptoken.intro-5>`__ of ISO/IEC IS
14882 (the C++ Standard) as follows:

   Calls to the functions ``request_stop``, ``stop_requested``, and ``stop_possible`` do not introduce data
   races **as if they were atomic operations**. […]

Modify `atomics.fences paragraph 2 through 4 <https://eel.is/c++draft/atomics.fences#2>`__ of ISO/IEC IS 14882 (the
C++ Standard) as follows:

   A release fence A synchronizes with an acquire fence B if there exist atomic operations X and Y, both operating on
   some atomic object M, such that A is sequenced before X, X modifies M, Y is sequenced before B, and Y reads the value
   written by X or a value written by any side effect in the hypothetical release sequence X would head if it were a
   release operation, **and each operation (A, B, X, and Y) specifies a scope that includes the thread that performed
   each other operation**.

   A release fence A synchronizes with an atomic operation B that performs an acquire operation on an atomic object M if
   there exists an atomic operation X such that A is sequenced before X, X modifies M, and B reads the value written by
   X or a value written by any side effect in the hypothetical release sequence X would head if it were a release
   operation, **and each operation (A, B, and X) specifies a scope that includes the thread that performed each other
   operation**.

   An atomic operation A that is a release operation on an atomic object M synchronizes with an acquire fence B if
   there exists some atomic operation X on M such that X is sequenced before B and reads the value written by A or a
   value written by any side effect in the release sequence headed by A, **and each operation (A, B, and X) specifies
   a scope that includes the thread that performed each other operation**.

.. _libcudacxx-extended-api-memory-model-message-passing:

Example: Message Passing
------------------------

The following example passes a message stored to the ``x`` variable by a
thread in block ``0`` to a thread in block ``1`` via the flag ``f``:

.. code:: cpp

   int x = 0;
   int f = 0;

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Thread 0 Block 0
     - Thread 0 Block 1
   * -
       .. code:: cpp

          x = 42;
          cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
          flag.store(1, memory_order_release);
     -
       .. code:: cpp

          cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
          while(flag.load(memory_order_acquire) != 1);
          assert(x == 42);

In the following variation of the previous example, two threads
concurrently access the ``f`` object without synchronization, which
leads to a **data race**, and exhibits **undefined behavior**:

.. code:: cpp

   int x = 0;
   int f = 0;

.. list-table::
   :widths: 50 50
   :header-rows: 1

   * - Thread 0 Block 0
     - Thread 0 Block 1
   * -
       .. code:: cpp

          x = 42;
          cuda::atomic_ref<int, cuda::thread_scope_block> flag(f);
          flag.store(1, memory_order_release); // UB: data race
     -
       .. code:: cpp

          cuda::atomic_ref<int, cuda::thread_scope_device> flag(f);
          while(flag.load(memory_order_acquire) != 1); // UB: data race
          assert(x == 42);

While the memory operations on ``f`` - the store and the loads - are
atomic, the scope of the store operation is “block scope”. Since the
store is performed by Thread 0 of Block 0, it only includes all other
threads of Block 0. However, the thread doing the loads is in Block 1,
i.e., it is not in a scope included by the store operation performed in
Block 0, causing the store and the load to not be “atomic”, and
introducing a data-race.

For more examples see the `PTX memory consistency model litmus
tests <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#axioms>`__.
