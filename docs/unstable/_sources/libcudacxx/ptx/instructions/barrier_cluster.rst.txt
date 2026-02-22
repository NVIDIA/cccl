.. _libcudacxx-ptx-instructions-barrier-cluster:

barrier.cluster
===============

-  PTX ISA:
   `barrier.cluster <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster>`__

Similar functionality is provided through the builtins
``__cluster_barrier_arrive(), __cluster_barrier_arrive_relaxed(), __cluster_barrier_wait()``,
as well as the ``cooperative_groups::cluster_group``
`API <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-group>`__.

The ``.aligned`` variants of the instructions are not exposed.

.. include:: generated/barrier_cluster.rst
