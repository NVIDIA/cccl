.. _stf_custom_data_interface:

Custom data interfaces
======================

CUDASTF can manage user-defined data types through custom data interfaces. A
data interface describes the shape of the data and implements the operations
CUDASTF needs to create, copy, and destroy data instances on different data
places.

A custom type used with the stream backend typically provides:

* A specialization of ``cuda::experimental::stf::shape_of<T>`` describing the
  logical dimensions of the data. Shapes used by ``parallel_for`` also provide
  an ``index_to_coords`` mapping.
* A class derived from ``stream_data_interface_simple<T>`` implementing
  ``stream_data_copy``, ``stream_data_allocate``, and
  ``stream_data_deallocate``. It may also implement ``pin_host_memory`` and
  ``unpin_host_memory``.
* A specialization of
  ``cuda::experimental::stf::streamed_interface_of<T>`` associating the user
  type with its stream data interface.
* A specialization of ``cuda::experimental::stf::hash<T>`` so instances can be
  identified by their layout and storage, independently of their contents.

The example below implements those customization points for a contiguous
two-dimensional ``matrix<T>``. It creates logical data from a host matrix,
updates it first with a regular CUDASTF task and then with ``parallel_for``, and
calls ``finalize()`` to write the result back to the original host allocation.

This is the canonical example built by the ``cudax.example.stf.custom_data_interface``
CMake target, so the documentation and compiled source remain synchronized.

.. literalinclude:: ../../../cudax/examples/stf/custom_data_interface.cu
   :language: cpp
   :caption: Complete custom data interface example

The example implements only the stream backend. Supporting ``graph_ctx`` also
requires a graph data interface that performs the corresponding allocation,
copy, and deallocation operations with CUDA Graph APIs.
