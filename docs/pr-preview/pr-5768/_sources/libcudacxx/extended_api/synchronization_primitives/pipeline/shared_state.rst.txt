
.. _libcudacxx-extended-api-synchronization-pipeline-pipeline-shared-state:

cuda::pipeline_shared_state
===============================

Defined in header ``<cuda/pipeline>``:

.. code:: cuda

   template <cuda::thread_scope Scope, cuda::std::uint8_t StagesCount>
   class cuda::pipeline_shared_state {
   public:
     __host__ __device__
     pipeline_shared_state();

     ~pipeline_shared_state() = default;

     pipeline_shared_state(pipeline_shared_state const&) = delete;

     pipeline_shared_state(pipeline_shared_state&&) = delete;
   };

The class template ``cuda::pipeline_shared_state`` is a storage type used to coordinate the threads participating
in a ``cuda::pipeline``.

.. rubric:: Template Parameters

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - ``Scope``
     - A :ref:`cuda::thread_scope <libcudacxx-extended-api-memory-model-thread-scopes>` denoting a scope including all
       the threads participating in the ``cuda::pipeline``. ``Scope`` cannot be ``cuda::thread_scope_thread``.
   * - ``StagesCount``
     - The number of stages for the *pipeline*.

.. rubric:: Member Functions

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Member function
     - Description
   * - ``(constructor)``
     - Constructs a ``cuda::pipeline_shared_state``.
   * - ``(destructor)`` [implicitly declared]
     - Destroys the ``cuda::pipeline_shared_state``.
   * - ``operator=`` [deleted]
     -  ``cuda::pipeline_shared_state`` is not assignable.

.. rubric:: Constructor

.. code:: cuda

   template <cuda::thread_scope Scope, cuda::std::uint8_t StagesCount>
   __host__ __device__
   cuda::pipeline_shared_state();

   template <cuda::thread_scope Scope, cuda::std::uint8_t StagesCount>
   cuda::pipeline_shared_state(cuda::pipeline_shared_state const&) = delete;

   template <cuda::thread_scope Scope, cuda::std::uint8_t StagesCount>
   cuda::pipeline_shared_state(cuda::pipeline_shared_state&&) = delete;

Construct a ``cuda::pipeline`` *shared state* object.

.. code:: cuda

   #include <cuda/pipeline>

   #pragma nv_diag_suppress static_var_with_dynamic_init

   __global__ void example_kernel() {
     __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> shared_state;
   }

`See it on Godbolt <https://godbolt.org/z/K4vKq4vd3>`__

.. rubric:: NVCC ``__shared__`` Initialization Warnings

When using libcu++ with NVCC, a ``__shared__`` ``cuda::pipeline_shared_state`` will lead to the following warning
because ``__shared__`` variables are not initialized:

.. code:: bash

   warning: dynamic initialization is not supported for a function-scope static
   __shared__ variable within a __device__/__global__ function

It can be silenced using ``#pragma nv_diag_suppress static_var_with_dynamic_init``.

.. rubric:: Example

.. code:: cuda

   #include <cuda/pipeline>

   // Disables `pipeline_shared_state` initialization warning.
   #pragma nv_diag_suppress static_var_with_dynamic_init

   __global__ void example_kernel(char* device_buffer, char* sysmem_buffer) {
     // Allocate a 2 stage block scoped shared state in shared memory.
     __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pss0;

     // Allocate a 2 stage block scoped shared state in device memory.
     auto* pss1 = new cuda::pipeline_shared_state<cuda::thread_scope_block, 2>;

     // Construct a 2 stage device scoped shared state in device memory.
     auto* pss2 =
       new (device_buffer) cuda::pipeline_shared_state<cuda::thread_scope_device, 2>;

     // Construct a 2 stage system scoped shared state in system memory.
     auto* pss3 =
       new (sysmem_buffer) cuda::pipeline_shared_state<cuda::thread_scope_system, 2>;
   }

`See it on Godbolt <https://godbolt.org/z/M9ah7r1Yx>`__
