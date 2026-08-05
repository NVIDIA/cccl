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

   pip install "cuda-cccl[cu13]"  # or "cuda-cccl[cu12]"

``cuda-cccl`` is a module-free metapackage. It installs ``cuda-compute``, which
provides the ``cuda.compute`` API and privately carries the CCCL headers it
needs, along with the ``cuda-toolkit`` pip packages for the chosen CUDA major
version.

To install only the implementation package, use the corresponding extra
directly:

.. code-block:: bash

   pip install "cuda-compute[cu13]"  # or "cuda-compute[cu12]"

.. important::

   When upgrading once from a former monolithic ``cuda-cccl`` release, first
   run ``python -m pip uninstall -y cuda-cccl``, then install the new
   metapackage. This avoids pip uninstalling files that moved to
   ``cuda-compute``. Later upgrades use the normal upgrade flow.

If you already have a CUDA toolkit installed on your system (e.g., via the
NVIDIA runfile, package manager, or Conda) and do not want pip to install it,
use the ``sysctk`` variants instead:

.. code-block:: bash

   pip install "cuda-cccl[sysctk13]"  # or "cuda-cccl[sysctk12]"

These install the same dependencies except ``cuda-toolkit``; it is your
responsibility to ensure a compatible CUDA toolkit is on ``PATH`` and
``LD_LIBRARY_PATH``.

For a minimal install without Numba (useful when you supply your own
:ref:`pre-compiled operators <cuda.compute.externally_compiled_operators>`), use:

.. code-block:: bash

   pip install "cuda-cccl[minimal-cu13]"      # pip-installed CUDA toolkit
   pip install "cuda-cccl[minimal-sysctk13]"  # system CUDA toolkit

Free-threaded Python support is currently validated with the ``minimal-cu12``
and ``minimal-cu13`` extras. The full ``cu12`` and ``cu13`` extras depend on
Numba CUDA and are not currently supported in free-threaded Python.

Install from conda-forge
~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively, you can install ``cuda-cccl`` using conda:

.. code-block:: bash

   conda install -c conda-forge cccl-python

This will install the CCCL Python libraries and their dependencies from the conda-forge channel.

Install from Source
~~~~~~~~~~~~~~~~~~~

For development or to access the latest ``cuda.compute`` features:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cccl.git
   cd cccl/python/cuda_cccl
   pip install -e ".[test-cu13]"  # or ".[test-cu12]", ".[test-sysctk13]", ".[test-sysctk12]"

The editable build reads libcudacxx, CUB, and Thrust headers from the canonical
repository directories. It does not copy them into ``python/cuda_cccl``.

The test extras do not install CuPy. To also run the CuPy-based
``cuda.compute`` examples, install CuPy separately, for example
``pip install cupy-cuda13x``.


Development Setup
~~~~~~~~~~~~~~~~~~

For contributing to ``cuda.compute`` or advanced development:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/NVIDIA/cccl.git
   cd cccl/python/cuda_cccl

   # Install in development mode with test dependencies
   pip install -e ".[test-cu13]"  # or ".[test-cu12]", ".[test-sysctk13]", ".[test-sysctk12]"

   # Run tests to verify everything works
   pytest tests/

Next Steps
----------

Now that you have ``cuda-cccl`` installed, check out:

* :doc:`compute/index` - Parallel computing primitives for operations on arrays or data ranges
