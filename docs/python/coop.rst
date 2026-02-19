.. _cccl-python-coop:

``cuda.coop``: Cooperative Algorithms
=====================================

The ``cuda.coop`` library provides cooperative primitives that operate at the
level of blocks and warps. It is designed to be used inside
`Numba CUDA kernels <https://nvidia.github.io/numba-cuda/>`_.

Quickstart
----------

Here is a minimal block-level reduction example:

.. literalinclude:: ../../python/cuda_cccl/tests/coop/examples/block/reduce.py
   :language: python
   :pyobject: custom_reduce_example
   :caption: Block-level reduction example.

Key ideas
---------

* **Collective operations** operate across a block or a warp and are called
  inside CUDA kernels.
* **Layouts** often use *blocked* or *striped* arrangements across threads.
* **Items per thread** must be known at compile time for most primitives.
* **Temp storage** can be managed explicitly when you want to share shared
  memory across multiple primitives.

.. _coop-flexible-data-arrangement:

Flexible data arrangement across threads
-----------------------------------------

Cooperative primitives operate on per-thread items that collectively form a
tile of data owned by a block or warp. Two common layouts are:

* **Blocked arrangement**: each thread owns a consecutive segment of items.
* **Striped arrangement**: each thread owns items that are interleaved across
  the tile.

Many block- and warp-level algorithms require a blocked arrangement because it
lets each thread process a contiguous slice of the tile while cooperating with
its neighbors. When choosing a layout, ensure that the number of items per
thread matches the primitive's expected arrangement.

Example collections
-------------------

For complete runnable examples and more advanced usage patterns, see our
full collection of `examples <https://github.com/NVIDIA/CCCL/tree/main/python/cuda_cccl/tests/coop/examples>`_.

Guides
------

.. toctree::
   :maxdepth: 2
   :caption: cuda.coop Guides

   coop_single_phase
   coop_two_phase
   coop_temp_storage
   coop_gpu_dataclass
   coop_advanced
   coop_faq
   coop_limitations
   coop_api
