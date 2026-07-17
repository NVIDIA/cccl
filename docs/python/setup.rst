.. _cccl-python-setup:

Setup and Installation
======================

This guide walks you through installing and setting up the CUDA Python Core Libraries (CCCL).

Prerequisites
-------------

Before installing cuda-cccl, ensure you have:

* **Python 3.10 or later**
* **CUDA Toolkit 12.x or 13.x**
* **Compatible NVIDIA GPU** with Compute Capability 7.5 or higher
* **Operating Systems:** Linux (tested on Ubuntu 20.04+) or Windows 10/11 (with WSL2 support)

Installation
------------

Install from PyPI
~~~~~~~~~~~~~~~~~

The easiest way to install ``cuda-cccl`` is using pip:

.. code-block:: bash

   pip install cuda-cccl[cu13]  # or cuda-cccl[cu12]

This will install ``cuda-cccl`` along with all required dependencies, including
the ``cuda-toolkit`` pip packages for the chosen CUDA major version.

If you already have a CUDA toolkit installed on your system (e.g., via the
NVIDIA runfile, package manager, or Conda) and do not want pip to install it,
use the ``sysctk`` variants instead:

.. code-block:: bash

   pip install cuda-cccl[sysctk13]  # or cuda-cccl[sysctk12]

These install the same dependencies except ``cuda-toolkit``; it is your
responsibility to ensure a compatible CUDA toolkit is on ``PATH`` and
``LD_LIBRARY_PATH``.

For a minimal install without Numba (useful when you supply your own
:ref:`pre-compiled operators <cuda.compute.externally_compiled_operators>`), use:

.. code-block:: bash

   pip install cuda-cccl[minimal-cu13]      # pip-installed CUDA toolkit
   pip install cuda-cccl[minimal-sysctk13]  # system CUDA toolkit

Free-threaded Python support is currently validated on Linux with the
``minimal-cu12`` and ``minimal-cu13`` extras. The full ``cu12`` and ``cu13``
extras depend on Numba CUDA and are not currently supported in free-threaded
Python.

Optional: Sequential Task Flow (``cuda-stf``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`cuda.stf._experimental <cccl-python-stf>` (CUDASTF) ships as a separate,
Linux-only package. Install it explicitly when you need it:

.. code-block:: bash

   pip install cuda-stf[cu13]  # or cuda-stf[cu12]

The ``cu12`` / ``cu13`` extras pull in a pip-installed CUDA toolkit plus Numba CUDA.
As with ``cuda-cccl``, ``cuda-stf`` also offers:

* ``sysctk12`` / ``sysctk13`` -- same as ``cu12`` / ``cu13`` but **without** the
  ``cuda-toolkit`` pip packages; you provide a compatible CUDA toolkit on ``PATH`` /
  ``LD_LIBRARY_PATH`` yourself.
* ``minimal-cu12`` / ``minimal-cu13`` -- CUDA bindings and toolkit only, **without**
  Numba (useful when you drive kernels through ``cuda.core`` / ``cuda.compute`` or your
  own launches).
* ``minimal-sysctk12`` / ``minimal-sysctk13`` -- minimal plus system-provided toolkit.

.. code-block:: bash

   pip install cuda-stf[sysctk13]        # system CUDA toolkit, with Numba
   pip install cuda-stf[minimal-cu13]    # pip CUDA toolkit, no Numba

Install ``cuda-cccl`` as well when using ``cuda.compute`` with STF or compiling
external C++ code that needs the libcudacxx, CUB, or Thrust headers.

Feature dependencies (installed separately as needed):

* ``cuda-cccl`` -- ``cuda.compute`` algorithms and C++ header discovery.
* ``numba`` / ``numba-cuda`` -- the Numba interop adapters (bundled by the non-minimal
  extras above).
* ``cupy`` -- some ``cuda.compute`` / interop examples.
* ``torch`` (PyTorch) -- the PyTorch interop adapter and its examples.
* ``warp-lang`` (NVIDIA Warp) -- the Warp interop examples.
* ``nvmath-python`` -- examples that call cuBLAS/cuSOLVER via nvmath.

Install ``cuda-stf`` from source (Linux only)::

   git clone https://github.com/NVIDIA/cccl.git
   cd cccl/python/cuda_stf
   pip install -e .[test-cu13]  # or .[test-cu12], .[test-sysctk13], .[test-sysctk12]

The ``test-*`` extras add ``cuda-cccl``, ``pytest``, ``pytest-xdist``, and CuPy so the
STF test suite (``pytest tests/``) can run. Building from source compiles the native
``cccl.c.experimental.stf`` / ``cudax`` extension, so a C++ toolchain and CMake
(``>=3.30``) with Ninja are required in addition to the CUDA toolkit.

Install from conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can install ``cuda-cccl`` using conda:

.. code-block:: bash

   conda install -c conda-forge cccl-python

This will install the CCCL Python libraries and their dependencies from the conda-forge channel.

Install from Source
~~~~~~~~~~~~~~~~~~~

For development or to access the latest features:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cccl.git
   cd cccl/python/cuda_cccl
   pip install -e .[test-cu13]  # or .[test-cu12], .[test-sysctk13], .[test-sysctk12]

The test extras do not install CuPy. To also run the CuPy-based
``cuda.compute`` examples, install CuPy separately, for example
``pip install cupy-cuda13x``.


Development Setup
~~~~~~~~~~~~~~~~~~

For contributing to cuda-cccl or advanced development:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/NVIDIA/cccl.git
   cd cccl/python/cuda_cccl

   # Install in development mode with test dependencies
   pip install -e .[test-cu13]  # or .[test-cu12], .[test-sysctk13], .[test-sysctk12]

   # Run tests to verify everything works
   pytest tests/

Next Steps
----------

Now that you have ``cuda-cccl`` installed, check out:

* :doc:`compute/index` - Parallel computing primitives for operations on arrays or data ranges
* :doc:`coop` - Block and warp-level cooperative algorithms for building custom CUDA kernels with Numba
* :doc:`stf` - Sequential Task Flow for CUDA (installed separately via ``cuda-stf``)
