mbarrier.arrive.noComplete.shared.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // mbarrier.arrive.noComplete.shared.b64                       state,  [addr], count;    // 5.  PTX ISA 70, SM_80
   template <typename=void>
   __device__ static inline uint64_t mbarrier_arrive_no_complete(
     uint64_t* addr,
     const uint32_t& count);
