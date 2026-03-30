.. _cccl-runtime-hierarchy:

.. |cuda_hierarchy| replace:: ``cuda::hierarchy``
.. _cuda_hierarchy: ../api/classcuda_1_1hierarchy.html

.. |cuda_make_hierarchy| replace:: ``cuda::make_hierarchy``
.. _cuda_make_hierarchy: ../api/namespacecuda_1a67bb05480718296ce6aff78859538637.html
.. |cuda_make_config| replace:: ``cuda::make_config``
.. _cuda_make_config: ../api/namespacecuda_1aa7b277627ddc60563f1818ae8e05ba2d.html
.. |cuda_grid_dims| replace:: ``cuda::grid_dims``
.. _cuda_grid_dims: ../api/namespacecuda_1a9b019989bfafbeec225ccfa07718216d.html
.. |cuda_cluster_dims| replace:: ``cuda::cluster_dims``
.. _cuda_cluster_dims: ../api/namespacecuda_1ad240665066f4a89a04af40e66e131ab7.html
.. |cuda_block_dims| replace:: ``cuda::block_dims``
.. _cuda_block_dims: ../api/namespacecuda_1a1649d0f7fed34582e19dba72f8c1b3d2.html
.. |cuda_warp| replace:: ``cuda::warp``
.. _cuda_warp: ../api/namespacecuda_1a25cebd54f74dcdc131654cb3977a1842.html
.. |cuda_gpu_thread| replace:: ``cuda::gpu_thread``
.. _cuda_gpu_thread: ../api/namespacecuda_1a1c4664dbad423f7bd37472020576c17c.html
.. |cuda_hierarchy_add_level| replace:: ``cuda::hierarchy_add_level``
.. _cuda_hierarchy_add_level: ../api/namespacecuda_1a2c197f19590504c7fccb5b0a9e8f361a.html
.. |cuda_get_launch_dimensions| replace:: ``cuda::get_launch_dimensions``
.. _cuda_get_launch_dimensions: ../api/namespacecuda_1a43e600724a8fbba0b8797014aa0246e9.html

Hierarchy
=========

The hierarchy API provides abstractions for representing and querying levels in the CUDA thread hierarchy (grid, cluster,
block, warp, and thread levels). It enables compile-time and runtime queries of thread dimensions and counts across
different hierarchy levels.

|cuda_hierarchy|_
---------------------------------------------------------------------
.. _cccl-runtime-hierarchy-hierarchy:

|cuda_hierarchy|_ is a type representing a hierarchy of CUDA threads. It combines hierarchy level descriptors
to represent dimensions of a (possibly partial) hierarchy. It supports accessing individual levels and queries
combining dimensions of multiple levels.

A hierarchy should be created using |cuda_make_hierarchy|_ rather than being constructed directly. The
hierarchy type can be used by itself, but its main purpose is to be part of a kernel launch configuration described
here: :ref:`Launch <cccl-runtime-launch>`. In that case, instead of calling |cuda_make_hierarchy|_, the same arguments
can be passed to |cuda_make_config|_.

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

|cuda_make_hierarchy|_
----------------------------------------------------------------------------------------------------
.. _cccl-runtime-hierarchy-make-hierarchy:

|cuda_make_hierarchy|_ creates a hierarchy from passed hierarchy level descriptors. Levels can be passed in
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

- |cuda_grid_dims|_ (compile-time and runtime overload forms)
- |cuda_cluster_dims|_ (compile-time and runtime overload forms)
- |cuda_block_dims|_ (compile-time and runtime overload forms)

Warp and thread levels are implicit and are queried via level objects (e.g., |cuda_warp|_,
|cuda_gpu_thread|_).

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
``cuda::block``, |cuda_warp|_, |cuda_gpu_thread|_):

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

|cuda_hierarchy_add_level|_
---------------------------------------------------------------------------------------------------------
.. _cccl-runtime-hierarchy-add-level:

|cuda_hierarchy_add_level|_ returns a new hierarchy that is a copy of the supplied hierarchy with a new level
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

|cuda_get_launch_dimensions|_
-----------------------------------------------------------------------------------------------------------
.. _cccl-runtime-hierarchy-launch-dimensions:

|cuda_get_launch_dimensions|_ returns a tuple of ``hierarchy_query_result`` objects containing dimensions from
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
