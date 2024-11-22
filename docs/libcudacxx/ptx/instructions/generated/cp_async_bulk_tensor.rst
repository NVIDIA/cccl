cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1a. PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename=void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3a. PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename=void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1b. PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename=void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3b. PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename=void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1c. PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename=void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3c. PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename=void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1d. PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename=void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3d. PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename=void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar];// 1e. PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename=void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // 3e. PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename=void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);
