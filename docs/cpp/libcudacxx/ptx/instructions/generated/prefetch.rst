..
   This file was automatically generated. Do not edit.

prefetch.global.L1
^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // prefetch.global.L1 [addr]; // PTX ISA 20, SM_50
   template <typename = void>
   __device__ static inline void prefetch_L1(
     const void* addr);

prefetch.global.L2
^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // prefetch.global.L2 [addr]; // PTX ISA 20, SM_50
   template <typename = void>
   __device__ static inline void prefetch_L2(
     const void* addr);

prefetch.global.L1::32B.valid_addr
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // prefetch.global.L1::32B.valid_addr [addr]; // PTX ISA 94, SM_90
   template <typename = void>
   __device__ static inline void prefetch_L1_32B(
     const void* addr);

prefetch.global.L2::evict_last
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // prefetch.global.L2::evict_last [addr]; // PTX ISA 74, SM_80
   template <typename = void>
   __device__ static inline void prefetch_L2_evict_last(
     const void* addr);

prefetch.global.L2::evict_normal
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // prefetch.global.L2::evict_normal [addr]; // PTX ISA 74, SM_80
   template <typename = void>
   __device__ static inline void prefetch_L2_evict_normal(
     const void* addr);

prefetch.tensormap
^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // prefetch.tensormap [addr]; // PTX ISA 80, SM_90
   template <typename = void>
   __device__ static inline void prefetch_tensormap(
     const void* addr);
