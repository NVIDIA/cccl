.. _cccl-python-setup:

Setup and Installation
======================

This guide walks you through installing and setting up the CUDA Core Compute
Libraries (CCCL) for Python.

Package layout
--------------

CCCL's Python functionality is published as three distributions:

* ``cuda-cccl`` is the aggregate metapackage. It contains no importable modules
  and installs ``cuda-compute``.
* ``cuda-compute`` provides the ``cuda.compute`` import package and depends on
  the same-version ``cccl-headers`` distribution.
* ``cccl-headers`` provides ``cuda.cccl.headers`` and the compatibility exports
  in ``cuda.cccl``. It bundles the CCCL C++ headers and relocatable CMake
  packages, but does not install a CUDA Toolkit or a Python compiler runtime.

Prerequisites
-------------

Before installing ``cuda.compute``, ensure you have:

* **Python 3.10 or later**
* **CUDA Toolkit 12.x or 13.x**
* **Compatible NVIDIA GPU** with Compute Capability 7.5 or higher
* **Operating Systems:** Linux (tested on Ubuntu 20.04+) or Windows 10/11 (with WSL2 support)

Installation
------------

Install the aggregate package from PyPI
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For most users, install the ``cuda-cccl`` aggregate package with the CUDA extra
matching your environment:

.. code-block:: bash

   pip install cuda-cccl[cu13]  # or cuda-cccl[cu12]

This installs ``cuda-compute``, ``cccl-headers``, and the ``cuda-toolkit`` pip
packages for the chosen CUDA major version.

If you already have a CUDA Toolkit installed on your system (for example, via
the NVIDIA runfile, a package manager, or Conda) and do not want pip to install
it, use a ``sysctk`` variant:

.. code-block:: bash

   pip install cuda-cccl[sysctk13]  # or cuda-cccl[sysctk12]

These install the same Python dependencies except ``cuda-toolkit``. You are
responsible for ensuring that a compatible CUDA Toolkit is available through
``PATH`` and, on Linux, ``LD_LIBRARY_PATH``.

For a minimal install without Numba, useful when supplying
:ref:`pre-compiled operators <cuda.compute.externally_compiled_operators>`, use:

.. code-block:: bash

   pip install cuda-cccl[minimal-cu13]      # pip-installed CUDA Toolkit
   pip install cuda-cccl[minimal-sysctk13]  # system CUDA Toolkit

Free-threaded Python support is currently validated with the ``minimal-cu12``
and ``minimal-cu13`` extras. The full ``cu12`` and ``cu13`` extras depend on
Numba CUDA and are not currently supported in free-threaded Python.

Upgrade from the former monolithic wheel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The former ``cuda-cccl`` wheel owned the ``cuda.compute`` and ``cuda.cccl``
files that are now owned by ``cuda-compute`` and ``cccl-headers``. Before the
first upgrade to the split layout, uninstall the old wheel in a separate
command so pip cannot remove the new owners' files while processing the old
wheel's RECORD:

.. code-block:: bash

   python -m pip uninstall -y cuda-cccl
   python -m pip install "cuda-cccl[cu13]"  # or the extra for your environment

This is a one-time migration. Subsequent upgrades within the three-wheel
layout can use the normal ``pip install --upgrade`` flow.

The deprecated ``cuda.cccl.parallel.experimental`` compatibility import is not
part of the split layout. Import the supported API directly from
``cuda.compute``.

Install individual distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install ``cuda-compute`` directly when you only need the ``cuda.compute`` API.
It installs its matching ``cccl-headers`` dependency automatically:

.. code-block:: bash

   pip install cuda-compute[cu13]  # or cuda-compute[cu12]

The ``sysctk*`` and ``minimal-*`` extras described above are also available on
``cuda-compute``.

For header-only development, install ``cccl-headers`` directly:

.. code-block:: bash

   pip install cccl-headers

The public Python entry point is ``cuda.cccl.headers``:

.. code-block:: python

   from cuda.cccl.headers import get_include_paths

   include_paths = get_include_paths()

Install from conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, install the aggregate CCCL Python package from conda-forge:

.. code-block:: bash

   conda install -c conda-forge cccl-python

Install from source
~~~~~~~~~~~~~~~~~~~

The three projects use exact same-version dependencies. Install the local
projects in dependency order so pip does not try to resolve an unreleased
sibling build from an index:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cccl.git
   cd cccl
   pip install -e ./python/cccl_headers
   pip install -e "./python/cuda_compute[test-cu13]"  # or test-cu12/test-sysctk*
   pip install -e ./python/cuda_cccl

The test extras do not install CuPy. To run the CuPy-based ``cuda.compute``
examples, install it separately, for example ``pip install cupy-cuda13x``.

Run the package tests from the repository root:

.. code-block:: bash

   pytest python/cccl_headers/tests
   pytest python/cuda_compute/tests
   pytest python/cuda_cccl/tests

Next Steps
----------

Now that you have the CCCL Python packages installed, see
:doc:`compute/index` for parallel computing primitives over arrays and data
ranges.
