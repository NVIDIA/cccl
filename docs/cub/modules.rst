CUB Modules
==================================================

.. toctree::
   :hidden:
   :maxdepth: 2

   thread_wide
   warp_wide
   block_wide
   device_wide


CUB provides state-of-the-art, reusable software components for every layer
of the CUDA programming model:

* **Parallel primitives**

  * :ref:`Thread-wide <thread-module>` primitives

    * Single thread reduction
    * Safely specialized for each underlying CUDA architecture

  * :ref:`Warp-wide <warp-module>` "collective" primitives

    * Cooperative warp-wide prefix scan, reduction, etc.
    * Safely specialized for each underlying CUDA architecture

  * :ref:`Block-wide <block-module>` "collective" primitives

    * Cooperative I/O, sort, scan, reduction, histogram, etc.
    * Compatible with arbitrary thread block sizes and types

  * :ref:`Device-wide <device-module>` primitives

    * Parallel sort, prefix scan, reduction, histogram, etc.
    * Compatible with CUDA dynamic parallelism
