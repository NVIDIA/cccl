.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Installation
============

``cuda-coop-tile`` supplies the Python frontend, C++ headers, and the shared
``cuda_coop_tile.tilelib`` provider. Use Python 3.11 through 3.15,
``cuda-tile==1.8.0``, and ``cuda-tile-preview==1.8.0``. Python kernels need
their Tile/SIMT foreign-call interface and CUDA 13.5 compiler libraries.

Build from source
-----------------

The Tile/SIMT interoperation SDK is a separate build input. Supply a compatible
SDK outside this checkout; its headers and sources are not included in the
source tree or Python wheel. The build compiles SDK support code into the
provider. The :doc:`/tile/development` guide describes the native build inputs.

From the CCCL repository root, build the wheel with explicit tool locations:

.. code-block:: bash

   python -m pip wheel --no-deps ./python/cuda_coop_tile -w dist \
       -Ccmake.define.CUDA_COOP_TILE_SDK_ROOT=/path/to/tile-sdk \
       -Ccmake.define.CUDA_COOP_TILE_INTERFACE_TRANSLATOR=/path/to/cuda-tile-translate \
       -Ccmake.define.CUDAToolkit_ROOT=/usr/local/cuda-13.5

Install the resulting wheel with the pinned frontend dependencies. When
building ``cuda-tile`` and ``cuda-tile-preview`` from source, use the matching
1.8.0 revision for both. The provider build does not build those
frontend packages.

.. code-block:: bash

   python -m pip install /path/to/cuda_coop_tile-VERSION-PLATFORM.whl
   python -m pip check

The examples use PyTorch for device arrays and CUDA streams. Install a
PyTorch build suitable for the selected CUDA environment before running them.
PyTorch is not part of the cooperative API.

Select a compiler toolkit
-------------------------

Use compatible CUDA headers, NVRTC and its builtins, TileIRAS, and nvJitLink.
Select the toolkit before starting Python. For a Linux toolkit installation:

.. code-block:: bash

   export CUDA_ROOT=/usr/local/cuda-13.5
   export CUDA_HOME="$CUDA_ROOT" CUDA_PATH="$CUDA_ROOT" CUDA_TOOLKIT_PATH="$CUDA_ROOT"
   export PATH="$CUDA_ROOT/bin:$PATH"
   export LD_LIBRARY_PATH="$CUDA_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

Keep ``CUDA_ROOT`` set because cuTile compiler launch paths may remove
``CUDA_HOME`` and ``CUDA_PATH`` from the compiler subprocess environment.

cuTile selects its compiler independently and can prefer an installed
TileIRAS Python package over ``PATH``. That compiler must be compatible with
the provider. The provider checks explicit toolkit roots and loaded NVRTC;
those checks alone do not validate cuTile's compiler selection or final linker.
Run the :doc:`programming` examples to exercise the complete compilation path.

Check the installation
----------------------

Run this outside the source checkout:

.. code-block:: python

   from importlib.metadata import version

   import cuda.coop.tile as coop

   for distribution in ("cuda-coop-tile", "cuda-tile", "cuda-tile-preview"):
       print(distribution, version(distribution))
   print("Tile module:", coop.__file__)
   print("C++ include directory:", coop.get_cpp_include_path())
   print("TileLib:", coop.get_cpp_tile_library_path())
   print("CMake package:", coop.get_cmake_dir())

An editable installation or ``PYTHONPATH`` can select a different source tree.
Successful imports locate the package; compiling and executing a kernel checks
the provider and compiler together.

Use these imports in Python kernel modules:

.. code-block:: python

   import cuda.coop.tile as coop
   import cuda.tile as ct

Accessing a primitive registers its cuTile lowering lazily. The host
configuration helpers remain available without loading the Python Tile compiler.

Development overrides
---------------------

``CUDA_COOP_TILE_LIBRARY_PATH`` selects an exact TileLib file. A missing
file or directory is an error. An unpackaged library also needs a nonempty
``CUDA_COOP_TILE_LIBRARY_VERSION`` that changes with the implementation or
its dependencies. Packaged artifacts use their manifest identity.
See :doc:`/tile/interop` for cache and identity behavior.

For provider headers, the Python frontend uses the first nonempty value of
``CUDA_COOP_CCCL_ROOT``, ``CCCL_ROOT``, or ``CCCL_DIR``. An invalid explicit root
is an error; see the :doc:`C++ installation rules </tile/cpp/installation>` for
accepted header roots and the distinction from CMake package directories.
