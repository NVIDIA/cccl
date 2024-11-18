.. _libcudacxx-setup-building:

Building & Testing libcu++
==========================

Step 0: Install Build Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a Bash shell:
================

.. code:: bash

   # Install LLVM (needed for LLVM's CMake modules)
   apt-get -y install llvm

   # Install CMake
   apt-get -y install cmake

   # Install the LLVM Integrated Tester (`lit`)
   apt-get -y install python-pip
   pip install lit

Windows, Native Build/Test
===========================

`Install Python <https://www.python.org/downloads/windows>`_.

Download `the get-pip.py bootstrap
script <https://bootstrap.pypa.io/get-pip.py>`_ and run it.

Install the LLVM Integrated Tester (``lit``) using a Visual Studio
command prompt:

.. code:: bash

   pip install lit

Step 1: Use the build scripts:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide build scripts for the most common test cases that should suit the majority of use cases.

In a Bash shell on Unix systems:

.. code:: bash

   ./ci/test_libcudacxx.sh

On Windows in ``x64 Native Tools Command Prompt``

.. code:: bash

   ./ci/windows/build_libcudacxx.ps1

This should cover most users needs, but offers less flexibility. Optionally one can directly invoke cmake

Optionally Step 2: Manually generate the Build Files:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a Bash shell or ``x64 Native Tools Command Prompt`` within the CCCL repository:

.. code:: bash

   cmake --preset libcudacxx-cpp17

Optionally Step 3: Build & Run the Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In a Bash shell or ``x64 Native Tools Command Prompt`` within the CCCL repository:

.. code:: bash

   cmake --build --preset libcudacxx-cpp17
   ctest --preset libcudacxx-lit-cpp17

This will build and run all available tests for that standard mode. However, usually only a subset of the full test
suite is relevant to a given PR. So rather than running all tests via ``ctest`` one can use lit to run a
subset of the test suite.

Optionally Step 4: Passing cmake options:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is also possible to pass individual options to the cmake configuration. For example

.. code:: bash

   cmake --preset libcudacxx-nvrtc-cpp17 -DCMAKE_CUDA_ARCHITECTURES="86"
   cmake --build --preset libcudacxx-cpp17
   ctest --preset libcudacxx-lit-cpp17

Optionally Step 5: Build a subset of the test suite:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   cd build
   lit libcudacxx-cpp17/RELATIVE_PATH_TO_TEST_OR_SUBFOLDER -sv

This will build and run all tests within ``RELATIVE_PATH_TO_TEST_OR_SUBFOLDER`` which must be a valid path within the CCCL.
Note that the name of the top level folder is the same as the name of the preset.

If only building the tests is desired one can pass ``-Dexecutor="NoopExecutor()"`` to the lit invocation.
.. code:: bash

   cd build
   lit libcudacxx-cpp17/RELATIVE_PATH_TO_TEST_OR_SUBFOLDER -sv -Dexecutor="NoopExecutor()"

Finally different standard modes can be tested by passing e.g ``--param=std=c++20``

NVRTC Build/Test:
=================

NVRTC tests can be build and tested the same way as the other tests

.. code:: bash

   cmake --preset libcudacxx-nvrtc-cpp17
   cmake --build --preset libcudacxx-cpp17
   ctest --preset libcudacxx-lit-cpp17

If you want to run individual tests its again

.. code:: bash

   cd build
   lit libcudacxx-nvrtc-cpp17/RELATIVE_PATH_TO_TEST_OR_SUBFOLDER -sv
