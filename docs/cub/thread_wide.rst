.. _thread-module:

Thread Primitives
==================================================

.. toctree::
   :glob:
   :hidden:
   :maxdepth: 2

   ${repo_docs_api_path}/*thread*

CUB thread-level algorithms are specialized for execution by a single CUDA thread:

* :cpp:func:`cub::ThreadReduce <cub::ThreadReduce>` computes reduction of items assigned to a single CUDA thread
