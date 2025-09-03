.. _libcudacxx-setup-building:

Building & Testing libcu++
==========================

libcu++ can be build and tested as shown in our `contributor guidelines <https://github.com/NVIDIA/cccl/blob/main/CONTRIBUTING.md#building-and-testing>`_.

However, often only a small subset of the full test suite needs to be run during development. For that we rely on ``lit``.
After libcu++ has been configured either through the build scripts or directly via a cmake preset one can then run.

.. code:: bash

   cd build
   lit libcudacxx-cpp17/RELATIVE_PATH_TO_TEST_OR_SUBFOLDER -sv

This will build and run all tests within ``RELATIVE_PATH_TO_TEST_OR_SUBFOLDER`` which must be a valid path within the CCCL.
Note that the name of the top level folder is the same as the name of the preset. For the build script the default is
``libcudacxx-cpp17``. As an example this is how to run all tests for ``cuda::std::span``, which are located in
``libcudacxx/test/libcudacxx/std/containers/views/views.span``

.. code:: bash

   cd build

   # Builds all tests within libcudacxx/test/libcudacxx/std/containers/views/views.span
   lit libcudacxx-cpp17/libcudacxx/test/libcudacxx/std/containers/views/views.span -sv

   # Builds the individual test array.pass.cpp
   lit libcudacxx-cpp17/libcudacxx/test/libcudacxx/std/containers/views/views.span/span.cons/array.pass.cpp -sv

If only building the tests and not running them is desired one can pass ``-Dexecutor="NoopExecutor()"`` to the lit invocation.
This is especially useful if the machine has no GPU or testing a different architecture

.. code:: bash

   cd build
   lit libcudacxx-cpp17/RELATIVE_PATH_TO_TEST_OR_SUBFOLDER -sv -Dexecutor="NoopExecutor()"

Finally different standard modes can be tested by passing e.g ``--param=std=c++20``

.. code:: bash

   cd build
   lit libcudacxx-cpp17/RELATIVE_PATH_TO_TEST_OR_SUBFOLDER -sv --param=std=c++20
