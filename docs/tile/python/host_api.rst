.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Host configuration helpers
==========================

These Python helpers locate the installed C++ headers, TileLib, and CMake
package. They are available without importing the Python Tile compiler.
Use them from the environment containing the
``cuda-coop-tile`` wheel. The :doc:`/tile/python/installation` guide describes the
artifact requirements; :doc:`/tile/cpp/programming` shows a complete CMake consumer.

.. currentmodule:: cuda.coop.tile

.. autoclass:: CppTileConfig
   :members: include_paths, nvcc_options

.. autofunction:: get_cpp_config

.. autofunction:: get_cpp_include_path

.. autofunction:: get_cpp_tile_library_path

.. autofunction:: get_cmake_dir

.. autofunction:: get_cpp_interface_path

Registration
------------

.. py:function:: register() -> None

   Register the Tile primitive lowerings with the cuTile compiler. Repeated
   calls return without registering a second time. Lowerings require TileIR
   bytecode version 13.5 or later and the compatible ``cuda-tile`` and
   ``cuda-tile-preview`` interfaces described in :doc:`/tile/python/installation`.

   Accessing a primitive through ``cuda.coop.tile`` already registers its
   compiler integration lazily, so ordinary kernels need no explicit call.
   Registration neither compiles a kernel nor verifies the installed
   TileLib or GPU. It imports the compiler integration and may raise an
   import error if that frontend is absent or incompatible.
