.. _cccl-python-setup:

Setup and Installation
======================

This guide walks you through installing and setting up the CUDA Python Core Libraries (CCCL).

Prerequisites
-------------

Before installing cuda-cccl, ensure you have:

* **Python 3.9 or later**
* **CUDA Toolkit 12.x or 13.x**
* **Compatible NVIDIA GPU** with Compute Capability 6.0 or higher
* **Operating Systems:** Linux (tested on Ubuntu 20.04+) or Windows 10/11 (with WSL2 support)

Installation
------------

Install from PyPI
~~~~~~~~~~~~~~~~~

The easiest way to install ``cuda-cccl`` is using pip:

.. code-block:: bash

   pip install cuda-cccl[cu13]  # or cuda-cccl[cu12]

This will install ``cuda-cccl`` along with all required dependencies.

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
   pip install -e .[test-cu13]  # or -e .[test-cu12]


Development Setup
~~~~~~~~~~~~~~~~~~

For contributing to cuda-cccl or advanced development:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/NVIDIA/cccl.git
   cd cccl/python/cuda_cccl

   # Install in development mode with test dependencies
   pip install -e .[test-cu13]  # or -e .[test-cu12]

   # Run tests to verify everything works
   pytest tests/

Next Steps
----------

Now that you have ``cuda-cccl`` installed, check out:

* :doc:`compute` - Parallel computing primitives for operations on arrays or data ranges
* :doc:`coop` - Block and warp-level cooperative algorithms for building custom CUDA kernels with Numba
