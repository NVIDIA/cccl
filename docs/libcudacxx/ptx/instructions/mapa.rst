.. _libcudacxx-ptx-instructions-mapa:

mapa
====

-  PTX ISA:
   `mapa <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mapa>`__

This instruction can `currently not be
implemented <https://github.com/NVIDIA/cccl/issues/1414>`__ by libcu++.
The instruction can be accessed through the cooperative groups
`cluster_group <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-group>`__
API:

Usage:
------

.. code:: cuda

   #include <cooperative_groups.h>

   __cluster_dims__(2)
   __global__ void kernel() {
       __shared__ int x;
       x = 1;
       namespace cg = cooperative_groups;
       cg::cluster_group cluster = cg::this_cluster();

       cluster.sync();

       // Get address of remote shared memory value:
       unsigned int other_block_rank = cluster.block_rank() ^ 1;
       int * remote_x = cluster.map_shared_rank(&bar, other_block_rank);

       // Write to remote value:
       *remote_x = 2;
   }
