.. _cccl-python-setup:

Setup and Installation
======================

This guide walks you through installing and setting up the CUDA Python Core Libraries (CCCL).

Prerequisites
-------------

Before installing cuda-cccl, ensure you have:

* **Python 3.9 or later**
* **CUDA Toolkit 12.x**
* **Compatible NVIDIA GPU** with Compute Capability 6.0 or higher
* **Operating Systems:** Linux (tested on Ubuntu 20.04+) or Windows 10/11 (with WSL2 support)

Installation
------------

Install from PyPI
~~~~~~~~~~~~~~~~~

The easiest way to install cuda-cccl is using pip:

.. code-block:: bash

   pip install cuda-cccl

This will install cuda-cccl along with all required dependencies including:

Install from Source
~~~~~~~~~~~~~~~~~~~

For development or to access the latest features:

.. code-block:: bash

   git clone https://github.com/NVIDIA/cccl.git
   cd cccl/python/cuda_cccl
   pip install -e .


Development Setup
~~~~~~~~~~~~~~~~~~

For contributing to cuda-cccl or advanced development:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/NVIDIA/cccl.git
   cd cccl/python/cuda_cccl

   # Install in development mode with test dependencies
   pip install -e .[test]

   # Run tests to verify everything works
   pytest tests/

Next Steps
----------

Now that you have ``cuda-cccl`` installed, check out:

* :doc:`parallel` - Device-level parallel algorithms
* :doc:`cooperative` - Block and warp-level cooperative primitives
