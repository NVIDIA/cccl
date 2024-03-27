.. _block-module:

Block-Wide "Collective" Primitives
==================================================

.. toctree::
   :glob:
   :hidden:
   :maxdepth: 2

   ${repo_docs_api_path}/*Block*

CUB block-level algorithms are specialized for execution by threads in the same CUDA thread block:

* :cpp:class:`cub::BlockAdjacentDifference <cub::BlockAdjacentDifference>` computes the difference between adjacent items partitioned across a CUDA thread block
* :cpp:class:`cub::BlockDiscontinuity <cub::BlockDiscontinuity>` flags discontinuities within an ordered set of items partitioned across a CUDA thread block
* :cpp:struct:`cub::BlockExchange <cub::BlockExchange>` rearranges data partitioned across a CUDA thread block
* :cpp:class:`cub::BlockHistogram <cub::BlockHistogram>` constructs block-wide histograms from data samples partitioned across a CUDA thread block
* :cpp:class:`cub::BlockLoad <cub::BlockLoad>` loads a linear segment of items from memory into a CUDA thread block
* :cpp:class:`cub::BlockMergeSort <cub::BlockMergeSort>` sorts items partitioned across a CUDA thread block
* :cpp:class:`cub::BlockRadixSort <cub::BlockRadixSort>` sorts items partitioned across a CUDA thread block using radix sorting method
* :cpp:struct:`cub::BlockReduce <cub::BlockReduce>` computes reduction of items partitioned across a CUDA thread block
* :cpp:class:`cub::BlockRunLengthDecode <cub::BlockRunLengthDecode>` decodes a run-length encoded sequence partitioned across a CUDA thread block
* :cpp:struct:`cub::BlockScan <cub::BlockScan>` computes a prefix scan of items partitioned across a CUDA thread block
* :cpp:struct:`cub::BlockShuffle <cub::BlockShuffle>` shifts items partitioned across a CUDA thread block
* :cpp:class:`cub::BlockStore <cub::BlockStore>` stores items partitioned across a CUDA thread block to a linear segment of memory
