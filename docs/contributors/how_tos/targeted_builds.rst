.. _infra-ci-targeted-builds:

Build and run targeted tests
============================

The ``ci/build_*.sh`` and ``ci/test_*.sh`` scripts build and run all headers,
tests, examples, etc for a single project. It is the right tool for reproducing
a CI job, but it is slow when you are iterating on a single test.

``ci/util/build_and_test_targets.sh`` builds and runs a named subset of CMake
targets against one preset. Use it to compile one test, run one CTest pattern,
or execute one libcudacxx lit test without rebuilding the rest of the project.
The full flag reference is in :doc:`/cccl/development/build_and_bisect_tools`.

Build a single CUB test
-----------------------

CCCL tests usually have a single name for their CMake target, ninja target, and CTest target.
It uniquely encodes the project, path, and test case, eg: ``cub.test.iterator``.

#. **Configure and build the target.** Pass the preset and the metatarget to
   ``--build-targets``::

     ci/util/build_and_test_targets.sh \
       --preset cub-cpp20 \
       --build-targets "cub.test.iterator"

   With no ``--ctest-targets``, the script configures and compiles, then stops.
   Compiling a test does not require a GPU.

#. **Run the target.** Add ``--ctest-targets`` with a CTest ``-R`` regex. The
   metatarget name works directly as the pattern::

     ci/util/build_and_test_targets.sh \
       --preset cub-cpp20 \
       --build-targets "cub.test.iterator" \
       --ctest-targets "cub.test.iterator"

   If ``--build-targets`` is omitted, the script assumes the targets are already built and
   skips to testing. Running tests may require a GPU.

Run a libcudacxx lit test
-------------------------

Some libcudacxx tests run under lit, not CTest. Pass lit test paths relative to
``libcudacxx/test/libcudacxx/``.

#. **Execute one lit test.** Use ``--lit-tests`` with the test path::

     ci/util/build_and_test_targets.sh \
       --preset libcudacxx \
       --lit-tests \
         "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp"

#. **Precompile without running.** Use ``--lit-precompile-tests`` to compile the
   test with a no-op executor. This catches compile errors without a GPU::

     ci/util/build_and_test_targets.sh \
       --preset libcudacxx \
       --lit-precompile-tests \
         "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp"

Run inside a devcontainer
-------------------------

To build against a specific CUDA toolkit and host compiler, wrap the invocation
with ``.devcontainer/launch.sh -d``. Valid CTK and host compiler values are
listed in the ``.devcontainer`` directory. Pass ``--gpus all`` when the run
needs a device::

  .devcontainer/launch.sh -d --cuda <CTK> --host <compiler> --gpus all -- \
    ci/util/build_and_test_targets.sh \
      --preset cub-cpp20 \
      --build-targets "cub.test.iterator" \
      --ctest-targets "cub.test.iterator"
