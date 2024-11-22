fence.proxy.async
^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.async; // 5. PTX ISA 80, SM_90
   template <typename=void>
   __device__ static inline void fence_proxy_async();

fence.proxy.async.global
^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.async{.space}; // 6. PTX ISA 80, SM_90
   // .space     = { .global, .shared::cluster, .shared::cta }
   template <cuda::ptx::dot_space Space>
   __device__ static inline void fence_proxy_async(
     cuda::ptx::space_t<Space> space);

fence.proxy.async.shared::cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.async{.space}; // 6. PTX ISA 80, SM_90
   // .space     = { .global, .shared::cluster, .shared::cta }
   template <cuda::ptx::dot_space Space>
   __device__ static inline void fence_proxy_async(
     cuda::ptx::space_t<Space> space);

fence.proxy.async.shared::cta
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // fence.proxy.async{.space}; // 6. PTX ISA 80, SM_90
   // .space     = { .global, .shared::cluster, .shared::cta }
   template <cuda::ptx::dot_space Space>
   __device__ static inline void fence_proxy_async(
     cuda::ptx::space_t<Space> space);
