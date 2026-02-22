.. _libcudacxx-ptx-instructions-mbarrier-arrive:

mbarrier.arrive
===============

-  PTX ISA:
   `mbarrier.arrive <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive>`__

.. _mbarrier.arrive-1:

mbarrier.arrive
---------------

Some of the listed PTX instructions below are semantically equivalent.
They differ in one important way: the shorter instructions are typically
supported on older compilers.

.. include:: generated/mbarrier_arrive.rst

mbarrier.arrive.no_complete
---------------------------

.. include:: generated/mbarrier_arrive_no_complete.rst

mbarrier.arrive.expect_tx
-------------------------

.. include:: generated/mbarrier_arrive_expect_tx.rst

Usage
-----

.. code:: cuda

   #include <cuda/ptx>
   #include <cuda/barrier>
   #include <cooperative_groups.h>

   __global__ void kernel() {
       using cuda::ptx::sem_release;
       using cuda::ptx::space_cluster;
       using cuda::ptx::space_shared;
       using cuda::ptx::scope_cluster;
       using cuda::ptx::scope_cta;

       using barrier_t = cuda::barrier<cuda::thread_scope_block>;
       __shared__ barrier_t bar;
       init(&bar, blockDim.x);
       __syncthreads();

       NV_IF_TARGET(NV_PROVIDES_SM_90, (
           // Arrive on local shared memory barrier:
           uint64_t token;
           token = cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_shared, &bar, 1);

           // Get address of remote cluster barrier:
           namespace cg = cooperative_groups;
           cg::cluster_group cluster = cg::this_cluster();
           unsigned int other_block_rank = cluster.block_rank() ^ 1;
           uint64_t * remote_bar = cluster.map_shared_rank(&bar, other_block_rank);

           // Sync cluster to ensure remote barrier is initialized.
           cluster.sync();

           // Arrive on remote cluster barrier:
           cuda::ptx::mbarrier_arrive_expect_tx(sem_release, scope_cluster, space_cluster, remote_bar, 1);
       )
   }
