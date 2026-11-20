.. _cccl-runtime-hierarchy:

.. |cuda_hierarchy| replace:: :ref:`cuda::hierarchy <libcudacxx-api-class-cuda-ns-hierarchy>`
.. |cuda_make_hierarchy| replace:: :ref:`cuda::make_hierarchy <libcudacxx-api-function-cuda-ns-make_hierarchy>`
.. |cuda_make_config| replace:: :ref:`cuda::make_config <libcudacxx-api-function-cuda-ns-make_config>`
.. |cuda_grid_dims| replace:: :ref:`cuda::grid_dims <libcudacxx-api-function-cuda-ns-grid_dims>`
.. |cuda_cluster_dims| replace:: :ref:`cuda::cluster_dims <libcudacxx-api-function-cuda-ns-cluster_dims>`
.. |cuda_block_dims| replace:: :ref:`cuda::block_dims <libcudacxx-api-function-cuda-ns-block_dims>`
.. |cuda_warp| replace:: :ref:`cuda::warp <libcudacxx-api-variable-cuda-ns-warp>`
.. |cuda_gpu_thread| replace:: :ref:`cuda::gpu_thread <libcudacxx-api-variable-cuda-ns-gpu_thread>`
.. |cuda_hierarchy_add_level| replace:: :ref:`cuda::hierarchy_add_level <libcudacxx-api-function-cuda-ns-hierarchy_add_level>`
.. |cuda_get_launch_dimensions| replace:: :ref:`cuda::get_launch_dimensions <libcudacxx-api-function-cuda-ns-get_launch_dimensions>`

Hierarchy
=========

The hierarchy API provides abstractions for representing and querying levels in the CUDA thread hierarchy (grid, cluster,
block, warp, and thread levels). It enables compile-time and runtime queries of thread dimensions and counts across
different hierarchy levels.

|cuda_hierarchy|
---------------------------------------------------------------------
.. _cccl-runtime-hierarchy-hierarchy:

|cuda_hierarchy| is a type representing a hierarchy of CUDA threads. It combines hierarchy level descriptors
to represent dimensions of a (possibly partial) hierarchy. It supports accessing individual levels and queries
combining dimensions of multiple levels.

A hierarchy should be created using |cuda_make_hierarchy| rather than being constructed directly. The
hierarchy type can be used by itself, but its main purpose is to be part of a kernel launch configuration described
here: :ref:`Launch <cccl-runtime-launch>`. In that case, instead of calling |cuda_make_hierarchy|, the same arguments
can be passed to |cuda_make_config|.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/hierarchy>

   auto h = cuda::make_hierarchy(
     cuda::grid_dims(256),
     cuda::block_dims<8, 8, 8>()
   );

   // Access level dimensions
   assert(h.level(cuda::grid).dims.x == 256);

   // Query counts across levels
   static_assert(cuda::gpu_thread.count(cuda::block, h) == 8 * 8 * 8);

|cuda_make_hierarchy|
----------------------------------------------------------------------------------------------------
.. _cccl-runtime-hierarchy-make-hierarchy:

|cuda_make_hierarchy| creates a hierarchy from passed hierarchy level descriptors. Levels can be passed in
ascending or descending order, and the function will automatically order them correctly.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/hierarchy>

   // Levels can be passed in any order
   auto h1 = cuda::make_hierarchy(
     cuda::grid_dims(256),
     cuda::cluster_dims<4>(),
     cuda::block_dims<8, 8, 8>()
   );

   auto h2 = cuda::make_hierarchy(
     cuda::block_dims<8, 8, 8>(),
     cuda::cluster_dims<4>(),
     cuda::grid_dims(256)
   );

   // Both create equivalent hierarchies
   static_assert(cuda::std::is_same_v<decltype(h1), decltype(h2)>);

Hierarchy Level Descriptors
----------------------------
.. _cccl-runtime-hierarchy-level-descriptors:

The hierarchy API provides level descriptor functions for grid, cluster, and block levels.
Each level supports both compile-time and runtime dimensions:

- |cuda_grid_dims| (compile-time and runtime overload forms)
- |cuda_cluster_dims| (compile-time and runtime overload forms)
- |cuda_block_dims| (compile-time and runtime overload forms)

Warp and thread levels are implicit and are queried via level objects (e.g., |cuda_warp|,
|cuda_gpu_thread|).

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/hierarchy>

   auto h = cuda::make_hierarchy(
     cuda::grid_dims(256, 128),      // Runtime grid dimensions
     cuda::cluster_dims<4>(),        // Compile-time cluster dimensions
     cuda::block_dims<32, 16>()      // Compile-time block dimensions
   );

Hierarchy Queries
-----------------
.. _cccl-runtime-hierarchy-queries:

Hierarchies support various query operations via level objects (``cuda::grid``, ``cuda::cluster``,
``cuda::block``, |cuda_warp|, |cuda_gpu_thread|):

- ``unit.count(level, hierarchy)`` - Count units within a level (e.g., threads per block)
- ``unit.rank(level, hierarchy)`` - Get the rank (linear index) of a unit within a level (device only)
- ``unit.dims(level, hierarchy)`` - Get dimensions of units within a level
- ``hierarchy.level<Level>()`` - Get the level descriptor for a specific level
- ``hierarchy.fragment<Unit, Level>()`` - Extract a fragment of the hierarchy

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/hierarchy>

   auto h = cuda::make_hierarchy(
     cuda::grid_dims(256),
     cuda::block_dims<8, 8, 8>()
   );

   // Get block-level descriptor
   auto block_desc = h.level(cuda::block);
   assert(block_desc.dims.x == 8);

   // Count threads per block
   static_assert(cuda::gpu_thread.count(cuda::block, h) == 512);

   // Get fragment (block to grid)
   auto fragment = h.fragment(cuda::block, cuda::grid);

|cuda_hierarchy_add_level|
---------------------------------------------------------------------------------------------------------
.. _cccl-runtime-hierarchy-add-level:

|cuda_hierarchy_add_level| returns a new hierarchy that is a copy of the supplied hierarchy with a new level
added. The function automatically determines whether to add the level at the top or bottom based on the existing
levels.

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/hierarchy>

   auto partial = cuda::make_hierarchy<cuda::block_level>(
     cuda::grid_dims(256),
     cuda::cluster_dims<4>()
   );

   auto complete = cuda::hierarchy_add_level(
     partial,
     cuda::block_dims<8, 8, 8>()
   );

|cuda_get_launch_dimensions|
-----------------------------------------------------------------------------------------------------------
.. _cccl-runtime-hierarchy-launch-dimensions:

|cuda_get_launch_dimensions| returns a tuple of ``hierarchy_query_result`` objects containing dimensions from
the hierarchy that can be used to launch kernels. The returned tuple has three elements if cluster_level is present
(grid, cluster, block dimensions), or two elements otherwise (grid, block dimensions).

Availability: CCCL 3.2.0 / CUDA 13.2

Example:

.. code:: cpp

   #include <cuda/hierarchy>

   auto h = cuda::make_hierarchy(
     cuda::grid_dims(256),
     cuda::cluster_dims<4>(),
     cuda::block_dims<8, 8, 8>()
   );

   auto [grid_dims, cluster_dims, block_dims] = cuda::get_launch_dimensions(h);
   // Can be used with cudaLaunchKernel or similar APIs
