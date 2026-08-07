..
   This file was automatically generated. Do not edit.

cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.multicast::cluster.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar], ctaMask; // PTX ISA 80, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar,
     const uint16_t& ctaMask);
