.. _cudax-graph:

Graphs library
==============

.. toctree::
   :glob:
   :maxdepth: 1

   api/struct*graph*

The headers of the graph library provide facilities to create and manage CUDA graphs.

This library is under construction and not yet ready for production use.

.. list-table::
   :widths: 25 45 30
   :header-rows: 0

   * - :ref:`<cuda/experimental/graph.cuh> <cudax-graph-graph-builder>`
     - Class `cuda::experimental::graph_builder`: An owning wrapper for a `cudaGraph_t` object.
     - cudax 2.9.0 / CCCL 2.9.0
   * - :ref:`<cuda/experimental/graph.cuh> <cudax-graph-graph-node-ref>`
     - Class `cuda::experimental::graph_node_ref`: A non-owning wrapper for a `cudaGraphNode_t` object.
     - cudax 2.9.0 / CCCL 2.9.0
   * - :ref:`<cuda/experimental/graph.cuh> <cudax-graph-graph>`
     - Class `cuda::experimental::graph`: An owning wrapper for a `cudaGraphExec_t` object.
     - cudax 2.9.0 / CCCL 2.9.0
