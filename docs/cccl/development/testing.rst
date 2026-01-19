.. _cccl-development-module-testing:

======================
CCCL Testing Utilities
======================

This document describes utilities provided for implementing the *internal* CCCL tests.
They are not intended to be used by end users, but for development of CCCL features only.
We reserve the right to change them at any time without warning.

----

-------------------------------------------------------------------
Test Variants: Generating Multiple Executables from a Single Source
-------------------------------------------------------------------

Some of CCCL's tests are very slow to build and are capable of exhausting RAM
during compilation/linking. To avoid such issues, large tests are split into
multiple executables to take advantage of parallel computation and reduce memory
usage.

CCCL facilitates this by providing a CMake-based solution for automatically generating multiple
test executables from a single source file. This is done by using one or more ``%PARAM%`` comments
in the test's source code, each of which defines a parameter that will be split across multiple
executables.

The CMake functions that implement this feature are in ``cmake/CCCLTestParams.cmake``.
An example of their usage is provided below.

Using ``%PARAM%``
-----------------

The ``%PARAM%`` hint provides an automated method of generating multiple test
executables from a single source file. To use it, add one or more special
comments to the test source file::

  // %PARAM% [definition] [label] [values]

CMake will parse the source file and extract these comments, using them to
generate multiple test executables for the full cartesian product of values.

- ``definition`` will be used as a preprocessor definition name. By convention,
  these begin with ``TEST_``.
- ``label`` is a short, human-readable label that will be used in the test
  executable's name to identify the test variant.
- ``values`` is a colon-separated list of values used during test generation. Only
  numeric values have been tested.

Example
*******

A source file containing the following hints::

  // %PARAM% TEST_FOO foo 0:1:2
  // %PARAM% TEST_LAUNCH lid 0:1

will generate six variants with unique preprocessor definitions:

+-----------------------------+-------------------------------------------+
| Executable Name             | Preprocessor Definitions                  |
+=============================+===========================================+
| ``<name_base>.foo_0.lid_0`` | ``-DTEST_FOO=0 -DTEST_LAUNCH=0 VAR_ID=0`` |
+-----------------------------+-------------------------------------------+
| ``<name_base>.foo_0.lid_1`` | ``-DTEST_FOO=0 -DTEST_LAUNCH=1 VAR_ID=1`` |
+-----------------------------+-------------------------------------------+
| ``<name_base>.foo_1.lid_0`` | ``-DTEST_FOO=1 -DTEST_LAUNCH=0 VAR_ID=2`` |
+-----------------------------+-------------------------------------------+
| ``<name_base>.foo_1.lid_1`` | ``-DTEST_FOO=1 -DTEST_LAUNCH=1 VAR_ID=3`` |
+-----------------------------+-------------------------------------------+
| ``<name_base>.foo_2.lid_0`` | ``-DTEST_FOO=2 -DTEST_LAUNCH=0 VAR_ID=4`` |
+-----------------------------+-------------------------------------------+
| ``<name_base>.foo_2.lid_1`` | ``-DTEST_FOO=2 -DTEST_LAUNCH=1 VAR_ID=5`` |
+-----------------------------+-------------------------------------------+

Changing ``%PARAM%`` Hints
**************************

Since CMake does not automatically reconfigure the build when source files are
modified, CMake will need to be rerun manually whenever the ``%PARAM%`` comments
change.

Using the CMake Variant Functions
---------------------------------

If using ``cccl_add_tests_from_src``, the variant parsing and test addition is
handled automatically, and the rest of this section can be skipped.

``cmake/CCCLTestParams.cmake`` provides the functions that implement parameter parsing.
See that file for detailed documentation. An example of their usage is:

..  code-block:: cmake

  set(test_src <path_to_source_file>)
  set(test_basename <test_name_derived_from_test_src>)

  cccl_detect_test_variants(${test_basename} "${test_src}")
  foreach (key IN LISTS variant_KEYS)
    add_executable(${${key}_NAME} "${test_src}")
    target_compile_definitions(${${key}_NAME} PRIVATE ${${key}_DEFINITIONS})
    add_test(NAME ${${key}_NAME} COMMAND ${${key}_NAME})
  endforeach()

Debugging
---------

Running CMake with ``--log-level=VERBOSE`` will print out extra information about
all detected test variants.

Additional Info
---------------

Ideally, only parameters that directly influence template instantiations
should be split out in this way. If changing a parameter doesn't change a
template type, the same template instantiations will be compiled into multiple
executables. This defeats the purpose of splitting up the test since the
compiler will generate redundant code across the new split executables.

The best candidate parameters for splitting are input value types, rather than
integral parameters like ``BLOCK_THREADS``, etc. Splitting by value type allows more
infrastructure (data generation, validation) to be reused. Splitting other
parameters can cause build times to increase since type-related infrastructure
has to be rebuilt for each test variant.
