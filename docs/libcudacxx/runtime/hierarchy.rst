.. _cccl-runtime-hierarchy:

Hierarchy
=========

The hierarchy API provides abstractions for representing and querying levels in the CUDA thread hierarchy (grid, cluster,
block, warp, and thread levels). It enables compile-time and runtime queries of thread dimensions and counts across
different hierarchy levels.

``cuda::hierarchy``
-------------------
.. _cccl-runtime-hierarchy-hierarchy:

``cuda::hierarchy`` is a type representing a hierarchy of CUDA threads. It combines hierarchy level descriptors to represent dimensions of a (possibly partial) hierarchy. It supports accessing individual levels and queries combining dimensions of multiple levels.

A hierarchy should be created using ``cuda::make_hierarchy()`` rather than being constructed directly. The hierarchy type can be used by itself, but its main purpose is to be part of a kernel launch configuration described here: :ref:`cccl-runtime-launch`. In that case, instead of calling ``cuda::make_hierarchy()``, the same arguments can be passed to ``cuda::make_config()``.

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

``cuda::make_hierarchy``
------------------------
.. _cccl-runtime-hierarchy-make-hierarchy:

``cuda::make_hierarchy()`` creates a hierarchy from passed hierarchy level descriptors. Levels can be passed in ascending or descending order, and the function will automatically order them correctly.

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

- ``cuda::grid_dims<x, y=1, z=1>()`` or ``cuda::grid_dims(x, y=1, z=1)``
- ``cuda::cluster_dims<x, y=1, z=1>()`` or ``cuda::cluster_dims(x, y=1, z=1)``
- ``cuda::block_dims<x, y=1, z=1>()`` or ``cuda::block_dims(x, y=1, z=1)``

Warp and thread levels are implicit and are queried via level objects (e.g., ``cuda::warp``, ``cuda::gpu_thread``).

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

Hierarchies support various query operations via level objects (``cuda::grid``, ``cuda::cluster``, ``cuda::block``,
``cuda::warp``, ``cuda::gpu_thread``):

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

``cuda::hierarchy_add_level``
------------------------------
.. _cccl-runtime-hierarchy-add-level:

``cuda::hierarchy_add_level()`` returns a new hierarchy that is a copy of the supplied hierarchy with a new level added. The function automatically determines whether to add the level at the top or bottom based on the existing levels.

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

``cuda::get_launch_dimensions``
--------------------------------
.. _cccl-runtime-hierarchy-launch-dimensions:

``cuda::get_launch_dimensions()`` returns a tuple of ``hierarchy_query_result`` objects containing dimensions from the hierarchy that can be used to launch kernels. The returned tuple has three elements if cluster_level is present (grid, cluster, block dimensions), or two elements otherwise (grid, block dimensions).

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
