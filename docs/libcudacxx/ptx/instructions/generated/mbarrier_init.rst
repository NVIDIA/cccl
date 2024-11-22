mbarrier.init.shared.b64
^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // mbarrier.init.shared.b64 [addr], count; // PTX ISA 70, SM_80
   template <typename=void>
   __device__ static inline void mbarrier_init(
     uint64_t* addr,
     const uint32_t& count);
