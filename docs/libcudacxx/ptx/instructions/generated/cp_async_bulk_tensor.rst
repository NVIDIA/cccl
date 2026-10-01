..
   This file was automatically generated. Do not edit.

cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 80, SM_90
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; // PTX ISA 80, SM_90
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_90
   // .dst       = { .shared::cta }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_90
   // .dst       = { .shared::cta }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_90
   // .dst       = { .shared::cta }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_90
   // .dst       = { .shared::cta }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_90
   // .dst       = { .shared::cta }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.bulk_group.override::global_address [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     uint64_t* smem_bar);

cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group.override::global_address.override::global_dim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.1d.dst.src.tile.bulk_group.override::global_address.override::global_dim [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[1],
     const int32_t (&tensorCoords)[1],
     const void* srcMem);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.bulk_group.override::global_address [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     uint64_t* smem_bar);

cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.2d.dst.src.tile.bulk_group.override::global_address.override::global_dim_stride [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[2],
     const B32 (&tensorLowerStrideToOverride)[1],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[2],
     const void* srcMem);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.bulk_group.override::global_address [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     uint64_t* smem_bar);

cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.3d.dst.src.tile.bulk_group.override::global_address.override::global_dim_stride [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[3],
     const B32 (&tensorLowerStrideToOverride)[2],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[3],
     const void* srcMem);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.bulk_group.override::global_address [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     uint64_t* smem_bar);

cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.4d.dst.src.tile.bulk_group.override::global_address.override::global_dim_stride [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[4],
     const B32 (&tensorLowerStrideToOverride)[3],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[4],
     const void* srcMem);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism [dstMem], [tensorMap, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address [dstMem], [tensorMap, gAddrToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.override::global_address
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.bulk_group.override::global_address [tensorMap, gAddrToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename = void>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::disabled.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::1.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::disabled.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80000000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8000.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::80.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_16bytes::8.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes.cta_group::2.mbarrier::report::validity::per_element::ff.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.mbarrier::complete_tx::bytes.cta_group.report_mechanism.override::global_address.override::global_dim_stride [dstMem], [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cta }
   // .src       = { .global }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   // .report_mechanism = { .mbarrier::report::disabled, .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, cuda::ptx::dot_cta_group Cta_Group, cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_shared_t,
     cuda::ptx::space_global_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     uint64_t* smem_bar);

cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group.override::global_address.override::global_dim_stride
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.tensor.5d.dst.src.tile.bulk_group.override::global_address.override::global_dim_stride [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride, tensorCoords], [srcMem]; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .global }
   // .src       = { .shared::cta }
   template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> = true>
   __device__ static inline void cp_async_bulk_tensor_override(
     cuda::ptx::space_global_t,
     cuda::ptx::space_shared_t,
     const void* tensorMap,
     const void* gAddrToOverride,
     const B16 (&tensorSizeToOverride)[5],
     const B32 (&tensorLowerStrideToOverride)[4],
     const B16& tensorUpperStrideToOverride,
     const int32_t (&tensorCoords)[5],
     const void* srcMem);
