.. _libcudacxx-ptx-instructions-mbarrier-init:

mbarrier.init
=============

-  PTX ISA:
   `mbarrier.arrive <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-init>`__

mbarrier.init.shared.b64
^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // mbarrier.init.shared.b64 [addr], count; // PTX ISA 70, SM_80
   template <typename=void>
   __device__ static inline void mbarrier_init(
     uint64_t* addr,
     const uint32_t& count);
