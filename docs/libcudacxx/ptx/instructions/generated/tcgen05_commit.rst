..
   This file was automatically generated. Do not edit.

tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster.b64 [smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_commit(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint64_t* smem_bar);

tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster.b64 [smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_commit(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint64_t* smem_bar);

tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [smem_bar], ctaMask; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_commit_multicast(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint64_t* smem_bar,
     uint16_t ctaMask);

tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [smem_bar], ctaMask; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_commit_multicast(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint64_t* smem_bar,
     uint16_t ctaMask);

tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.multicast::cluster::32b.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster.multicast::cluster::32b.b64 [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_commit_multicast_32b(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint64_t* smem_bar,
     uint32_t ctaMask);

tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster::32b.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.commit.cta_group.mbarrier::arrive::one.shared::cluster.multicast::cluster::32b.b64 [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_commit_multicast_32b(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint64_t* smem_bar,
     uint32_t ctaMask);

tcgen05.commit.cta_group::1.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.commit.cta_group.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.b64 [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_commit_sync_restrict_shared_read_mma_a(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint64_t* smem_bar);

tcgen05.commit.cta_group::2.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.commit.cta_group.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.b64 [smem_bar]; // PTX ISA 94, SM_107a, SM_107f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_commit_sync_restrict_shared_read_mma_a(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint64_t* smem_bar);

tcgen05.commit.cta_group::1.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.multicast::cluster::32b.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.commit.cta_group.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.multicast::cluster::32b.b64 [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_commit_sync_restrict_shared_read_mma_a_multicast_32b(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint64_t* smem_bar,
     uint32_t ctaMask);

tcgen05.commit.cta_group::2.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.multicast::cluster::32b.b64
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.commit.cta_group.mbarrier::arrive::one.sync_restrict::shared::read::mma::a.shared::cluster.multicast::cluster::32b.b64 [smem_bar], ctaMask; // PTX ISA 94, SM_107a, SM_107f
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_commit_sync_restrict_shared_read_mma_a_multicast_32b(
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint64_t* smem_bar,
     uint32_t ctaMask);
