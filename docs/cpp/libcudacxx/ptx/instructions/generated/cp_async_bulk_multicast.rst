..
   This file was automatically generated. Do not edit.

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.multicast::cluster [dstMem], [srcMem], size, [smem_bar], ctaMask; // PTX ISA 80, SM_90a, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* srcMem,
     const uint32_t& size,
     uint64_t* smem_bar,
     const uint16_t& ctaMask);

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.multicast::cluster::32b [dstMem], [srcMem], size, [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   template <typename = void>
   __device__ static inline void cp_async_bulk_multicast_32b(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     void* dstMem,
     const void* srcMem,
     const uint32_t& size,
     uint64_t* smem_bar,
     const uint32_t& ctaMask);

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_16bytes::80000000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.multicast::cluster::32b.report_mechanism [dstMem], [srcMem], size, [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .report_mechanism = { .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_multicast_32b(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* srcMem,
     const uint32_t& size,
     uint64_t* smem_bar,
     const uint32_t& ctaMask);

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_16bytes::8000
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.multicast::cluster::32b.report_mechanism [dstMem], [srcMem], size, [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .report_mechanism = { .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_multicast_32b(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* srcMem,
     const uint32_t& size,
     uint64_t* smem_bar,
     const uint32_t& ctaMask);

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_16bytes::80
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.multicast::cluster::32b.report_mechanism [dstMem], [srcMem], size, [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .report_mechanism = { .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_multicast_32b(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* srcMem,
     const uint32_t& size,
     uint64_t* smem_bar,
     const uint32_t& ctaMask);

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_16bytes::8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.multicast::cluster::32b.report_mechanism [dstMem], [srcMem], size, [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .report_mechanism = { .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_multicast_32b(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* srcMem,
     const uint32_t& size,
     uint64_t* smem_bar,
     const uint32_t& ctaMask);

cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster::32b.mbarrier::report::validity::per_element::ff
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // cp.async.bulk.dst.src.mbarrier::complete_tx::bytes.multicast::cluster::32b.report_mechanism [dstMem], [srcMem], size, [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .dst       = { .shared::cluster }
   // .src       = { .global }
   // .report_mechanism = { .mbarrier::report::validity::per_16bytes::80000000, .mbarrier::report::validity::per_16bytes::8000, .mbarrier::report::validity::per_16bytes::80, .mbarrier::report::validity::per_16bytes::8, .mbarrier::report::validity::per_element::ff }
   template <cuda::ptx::dot_report_mechanism Report_Mechanism>
   __device__ static inline void cp_async_bulk_multicast_32b(
     cuda::ptx::space_cluster_t,
     cuda::ptx::space_global_t,
     cuda::ptx::report_mechanism_t<Report_Mechanism> report_mechanism,
     void* dstMem,
     const void* srcMem,
     const uint32_t& size,
     uint64_t* smem_bar,
     const uint32_t& ctaMask);
