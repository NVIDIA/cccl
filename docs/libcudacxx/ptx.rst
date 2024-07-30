.. _libcudacxx-ptx:

PTX
=====


The ``cuda::ptx`` namespace contains functions that map one-to-one to
`PTX instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>`__.
These can be used for maximal control of the generated code, or to
experiment with new hardware features before a high-level C++ API is
available.

.. toctree::
   :maxdepth: 1

   ptx/examples
   ptx/instructions

Versions and compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``cuda/ptx`` header is intended to present a stable API within one major
version of the CTK on a best effort basis. This means that:

-  All functions are marked static inline.

-  The type of a function parameter can be changed to be more generic if
   that means that code that called the original version can still be
   compiled.

-  Good exposure of the PTX should be high priority. If, at a new major
   version, we face a difficult choice between breaking
   backward-compatibility and an improvement of the PTX exposure, we
   will tend to the latter option more easily than in other parts of
   libcu++.

The API does not guarantee stability of template parameters. The order and
number of template parameters may change. Use arguments to driver overload
resolution as in the code below to ensure forward-compatibility:

.. code:: cuda

   // Use arguments to drive overload resolution:
   cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &bar, 1);

   // Specifying templates directly is not forward-compatible, as order and number
   // of template parameters may change in a minor release:
   cuda::ptx::mbarrier_arrive_expect_tx<cuda::ptx::sem_release_t>(
     cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &bar, 1
   );

**PTX ISA version and compute capability.** Each binding notes under
which PTX ISA version and SM version it may be used. Example:

.. code:: cuda

   // mbarrier.arrive.shared::cta.b64 state, [addr]; // 1.  PTX ISA 70, SM_80
   __device__ inline uint64_t mbarrier_arrive(
     cuda::ptx::sem_release_t sem,
     cuda::ptx::scope_cta_t scope,
     cuda::ptx::space_shared_t space,
     uint64_t* addr);

To check if the current compiler is recent enough, use:

.. code:: cuda

   #if __cccl_ptx_isa >= 700
   cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &bar, 1);
   #endif

Ensure that you only call the function when compiling for a recent
enough compute capability (SM version), like this:

.. code:: cuda

   NV_IF_TARGET(NV_PROVIDES_SM_80,(
     cuda::ptx::mbarrier_arrive(cuda::ptx::sem_release, cuda::ptx::scope_cta, cuda::ptx::space_shared, &bar, 1);
   ));

For more information on which compilers correspond to which PTX ISA, see
the `PTX ISA release notes <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#release-notes>`__.
