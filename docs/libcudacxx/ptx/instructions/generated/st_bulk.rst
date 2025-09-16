..
   This file was automatically generated. Do not edit.

st.bulk.weak.shared::cta
^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // st.bulk.weak.shared::cta [addr], size, initval; // PTX ISA 86, SM_100
   template <int N32>
   __device__ static inline void st_bulk(
     void* addr,
     uint64_t size,
     cuda::ptx::n32_t<N32> initval);
