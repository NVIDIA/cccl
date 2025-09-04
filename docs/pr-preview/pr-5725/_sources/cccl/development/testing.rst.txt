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

``cmake/CCCLTestParams.cmake`` provides the functions that implement this functionality.
See that file for detailed documentation. An example of their usage is:

..  code-block:: cmake

  set(test_src <path_to_source_file>)
  set(test_name <test_name_derived_from_test_src>)

  # Parse %PARAM% comments from the source file and generate lists of labels/definitions:
  cccl_parse_variant_params("${test_src}" num_variants variant_labels variant_defs)

  if (num_variants EQUAL 0)
    # Add test with no variants named `test_name` here. Example:
    add_executable("${test_name}" "${test_src}")
    add_test(NAME "${test_name}" COMMAND "${test_name}")
  else() # Has variants:
    # Optional: log the detected variant info to CMake's VERBOSE output stream:
    cccl_log_variant_params("${test_name}" ${num_variants} variant_labels variant_defs)

    # Subtract 1 to support the inclusive endpoint of foreach(...RANGE...):
    math(EXPR var_range_end "${num_variants} - 1")
    foreach(var_idx RANGE ${var_range_end})
      # Get the variant label and definitions for the current index:
      cccl_get_variant_data(variant_labels variant_defs ${var_idx} var_label var_defs)
      set(var_name "${test_name}.${var_label}")

      # Add the test with the current variant label and definitions.
      # Example:
      add_executable("${var_name}" "${test_src}")
      target_compile_definitions("${var_name}" PRIVATE ${var_defs})
      add_test(NAME "${var_name}" COMMAND "${var_name}")
    endforeach()
  endif()

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
