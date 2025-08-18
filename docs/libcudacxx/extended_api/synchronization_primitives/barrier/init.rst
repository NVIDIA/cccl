.. _libcudacxx-extended-api-synchronization-barrier-barrier-init:

cuda::barrier::init
=======================

Defined in header ``<cuda/barrier>``:

.. code:: cuda

   template <cuda::thread_scope Scope,
             typename CompletionFunction = /* unspecified */>
   class barrier {
   public:
     // ...

     __host__ __device__
     friend void init(cuda::std::barrier* bar,
                      cuda::std::ptrdiff_t expected,
                      CompletionFunction cf = CompletionFunction{});
   };

The friend function ``cuda::barrier::init`` may be used to initialize an
:ref:`cuda::barrier <libcudacxx-extended-api-synchronization-barrier>` that has not been initialized.

When using libcu++ with NVCC, ``__shared__`` ``cuda::barrier`` will not have its constructors run because ``__shared__``
variables are not initialized. ``cuda::barrier::init`` should be used to properly initialize such a
:ref:`cuda::barrier <libcudacxx-extended-api-synchronization-barrier>`.

An NVCC diagnostic warning about the ignored constructor will be emitted:

.. code:: bash

   warning: dynamic initialization is not supported for a function-scope static
   __shared__ variable within a __device__/__global__ function

It can be silenced using ``#pragma nv_diag_suppress static_var_with_dynamic_init``.

Example
-------

.. code:: cuda

   #include <cuda/barrier>

   // Disables `cuda::barrier` initialization warning.
   #pragma nv_diag_suppress static_var_with_dynamic_init

   __global__ void example_kernel() {
     __shared__ cuda::barrier<cuda::thread_scope_block> bar;
     init(&bar, 1);
   }

`See it on Godbolt <https://godbolt.org/z/nK5q3xh34>`_
