.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Installation
============

The C++ frontend consists of ``<cuda/coop/tile.cuh>`` and the shared
``cuda_coop_tile.tilelib`` provider. Build and install them with CMake;
a Python package installation is not required.

Build inputs
------------

Use CMake 3.25.2 or newer for C++ consumer and test builds. Building the
provider or Python package requires CMake 3.21 or newer.

Use CUDA 13.5 with Tile/SIMT interoperation support and the separately supplied
Tile/SIMT interoperation SDK headers and sources. Build ``cuda-tile-translate``
from the public `CUDA Tile v13.5.0 release
<https://github.com/NVIDIA/cuda-tile/releases/tag/v13.5.0>`_ matching CUDA 13.5;
the declaration build requires TileIR bytecode version 13.5.
Keep these inputs outside this checkout.
See :doc:`/tile/development` for the native source and declaration build.

From the repository root:

.. code-block:: bash

   cmake -S tile -B build/tile -G Ninja \
       -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_INSTALL_PREFIX=/path/to/tile-install \
       -DCUDAToolkit_ROOT=/usr/local/cuda-13.5 \
       -DCUDA_COOP_TILE_SDK_ROOT=/path/to/tile-sdk \
       -DCUDA_COOP_TILE_INTERFACE_TRANSLATOR=/path/to/cuda-tile-translate
   cmake --build build/tile --target cuda_coop_tile_library
   cmake --install build/tile

The SDK inputs are required for building the provider, not for compiling an
application against the installed headers and TileLib. Keep the installed
provider and consumer compiler compatible.

Consume the installation
------------------------

The :doc:`programming` includes a complete CMake consumer. Pass the installation
prefix through ``CMAKE_PREFIX_PATH`` and select CUDA 13.5's ``nvcc`` explicitly.
The package exports ``cuda-coop-tile::tile`` and
``cuda_coop_tile_link(target)``. The helper supplies include paths and Tile
compiler flags and tracks the TileLib as a compilation dependency.

On Linux, the dynamic loader must be able to find the matching CUDA 13.5
``libnvrtc`` library when the compiler loads the provider. A toolkit configured
in the system library search path needs no extra environment setting. For a
toolkit installed elsewhere, add its library directory before compiling:

.. code-block:: bash

   export LD_LIBRARY_PATH="/path/to/cuda-13.5/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

.. code-block:: cpp

   #include <cuda/coop/tile.cuh>
   namespace coop = ::cuda::coop::tile;

The Python distribution also bundles the C++ headers, TileLib, and CMake
package. Applications using that distribution can locate its CMake package
through the optional
:doc:`Python host helpers </tile/python/host_api>`.

``CUDA_COOP_TILE_LIBRARY_PATH`` can select an exact replacement TileLib.
The selected file must match the frontend declarations and compiler ABI.
See :doc:`/tile/interop` for artifact identity and rebuild behavior.

The provider uses its installed CCCL header bundle by default. To override the
headers used for its generated CUB implementations, prefer
``CUDA_COOP_CCCL_ROOT``. Its header resolver uses the first nonempty environment
variable among ``CUDA_COOP_CCCL_ROOT``, ``CCCL_ROOT``, and ``CCCL_DIR``. Each
must name a CCCL source checkout or a header bundle containing ``cub``,
``thrust``, and ``cuda`` directories, directly or under ``include``. An invalid
explicit root produces an error.

These aliases apply to the provider's header resolution. In particular, an
exported ``CCCL_DIR`` naming a CMake package configuration directory is not a
header root. Unset such generic overrides to use the installed bundle, or set
``CUDA_COOP_CCCL_ROOT`` to the intended headers. The CMake target supplies the
C++ frontend's include paths separately.
