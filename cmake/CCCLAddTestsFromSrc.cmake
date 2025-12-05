# cccl_add_tests_from_src(test_names_var test_src
#   [FUNC_IS_SRC_SUPPORTED <function>]
#   [FUNC_CUSTOMIZE_SRC <function>]
#   [FUNC_SRC_TO_BASENAME <function>]
#   [FUNC_IS_TEST_SUPPORTED <function>]
#   [FUNC_IS_COMPILE_FAIL_TEST <function>]
#   [FUNC_CUSTOMIZE_EXECUTABLE <function>]
#   [FUNC_CUSTOMIZE_TEST_COMMAND <function>]
#   [FUNC_CUSTOMIZE_TEST_PROPERTIES <function>]
#   [EXTRA_ARGN [ARG1 [ARG2 ...]]])
# )
#
# Implements common testing features for CCCL projects, including:
# - Detecting %PARAM% variants in test source files
# - Adding compile-fail tests via cccl_add_xfail_compile_target_test
#
# Function hooks may be provided to customize various aspects of test configuration:
#
# - FUNC_IS_SRC_SUPPORTED: function(out_var test_src)
#   Determines if the given source file should be processed for tests.
#   Default: always TRUE.
# - FUNC_CUSTOMIZE_SRC: function(out_var test_src)
#   Customizes the test source file (e.g. wrapping .cu in .cpp for non-CUDA backends).
#   Set out_var to the new test source name that will be used to compile the test.
#   Note that the original test source file will continue to be used everywhere other than
#   add_executable. This keeps parsing logic simple and cache-friendly.
#   Default: no customization.
# - FUNC_SRC_TO_BASENAME: function(out_var test_src)
#   Converts the source file path to a base name for use in test names and metatarget paths.
#   Default: uses the filename without extension.
# - FUNC_IS_TEST_SUPPORTED: function(out_var test_src test_name)
#   Determines if the given test (source + test case variant name) should be added.
#   Default: always TRUE.
# - FUNC_IS_COMPILE_FAIL_TEST: function(out_var test_src test_name)
#   Determines if the given test is a compile-fail test.
#   Default: TRUE if the test name contains ".fail" or "_fail".
# - FUNC_CUSTOMIZE_EXECUTABLE: function(test_src test_name test_executable_target)
#   Customizes the test executable target (e.g. adding libraries, defs, options, etc).
#   Default: links cccl.compiler_interface.
# - FUNC_CUSTOMIZE_TEST_COMMAND: function(out_var test_src test_name test_executable_target)
#   Customizes the test command to be used in add_test.
#   Default: Executes the test executable file.
# - FUNC_CUSTOMIZE_TEST_PROPERTIES: function(test_src test_name test_executable_target ctest_name)
#   Customizes the CTest test properties (e.g. SKIP_REGULAR_EXPRESSION, WILL_FAIL, etc).
#   Default: skips tests that print CCCL_SKIP_TEST.
#
# If provided, the EXTRA_ARGN arguments are passed to each of the customization functions.
#
# The resulting list of added test names is returned in test_names_var.
# Further documentation and examples are provided in docs/cccl/development/testing.rst.
function(cccl_add_tests_from_src test_names_var test_src)
  set(options)
  set(
    oneValueArgs
    FUNC_IS_SRC_SUPPORTED
    FUNC_CUSTOMIZE_SRC
    FUNC_SRC_TO_BASENAME
    FUNC_IS_TEST_SUPPORTED
    FUNC_IS_COMPILE_FAIL_TEST
    FUNC_CUSTOMIZE_EXECUTABLE
    FUNC_CUSTOMIZE_TEST_COMMAND
    FUNC_CUSTOMIZE_TEST_PROPERTIES
  )
  set(multiValueArgs EXTRA_ARGN)
  cmake_parse_arguments(
    self
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )
  cccl_parse_arguments_error_checks(
    "cccl_add_tests_from_src"
    ERROR_UNPARSED
    DEFAULT_VALUES
      FUNC_IS_SRC_SUPPORTED "_cccl_default_is_src_supported"
      FUNC_CUSTOMIZE_SRC "_cccl_default_customize_src"
      FUNC_SRC_TO_BASENAME "_cccl_default_src_to_basename"
      FUNC_IS_TEST_SUPPORTED "_cccl_default_is_test_supported"
      FUNC_IS_COMPILE_FAIL_TEST "_cccl_default_is_compile_fail_test"
      FUNC_CUSTOMIZE_EXECUTABLE "_cccl_default_customize_executable"
      FUNC_CUSTOMIZE_TEST_COMMAND "_cccl_default_customize_test_command"
      FUNC_CUSTOMIZE_TEST_PROPERTIES "_cccl_default_customize_test_properties"
  )
  set(is_src_supported)
  cmake_language(
    CALL ${self_FUNC_IS_SRC_SUPPORTED}
    is_src_supported
    "${test_src}"
    ${self_EXTRA_ARGN}
  )
  if (NOT is_src_supported)
    set(${test_names_var} "" PARENT_SCOPE)
    return()
  endif()

  set(orig_test_src "${test_src}")
  cmake_language(
    CALL ${self_FUNC_CUSTOMIZE_SRC}
    test_src
    "${test_src}"
    ${self_EXTRA_ARGN}
  )

  cmake_language(
    CALL ${self_FUNC_SRC_TO_BASENAME}
    test_basename
    "${orig_test_src}"
    ${self_EXTRA_ARGN}
  )

  # Check for %PARAM% variants. Use the original source for parsing to improve caching:
  cccl_detect_test_variants(${test_basename} "${orig_test_src}")
  set(test_names)
  foreach (key IN LISTS variant_KEYS)
    set(test_name "${${key}_NAME}")
    set(test_defs "${${key}_DEFINITIONS}")

    set(is_test_supported)
    cmake_language(
      CALL ${self_FUNC_IS_TEST_SUPPORTED}
      is_test_supported
      "${orig_test_src}"
      ${test_name}
      ${self_EXTRA_ARGN}
    )
    if (NOT is_test_supported)
      continue()
    endif()

    cmake_language(
      CALL ${self_FUNC_IS_COMPILE_FAIL_TEST}
      is_compile_fail_test
      "${orig_test_src}"
      ${test_name}
      ${self_EXTRA_ARGN}
    )

    set(test_executable_target "${test_name}")
    cccl_add_executable(
      ${test_executable_target}
      SOURCES "${test_src}"
      NO_METATARGETS # Added manually for non-fail tests
    )
    target_compile_definitions(${test_executable_target} PRIVATE ${test_defs})
    cmake_language(
      CALL ${self_FUNC_CUSTOMIZE_EXECUTABLE}
      "${orig_test_src}"
      ${test_name}
      ${test_executable_target}
      ${self_EXTRA_ARGN}
    )

    set(ctest_name "${test_name}")
    if (is_compile_fail_test)
      cccl_add_xfail_compile_target_test(
        ${test_executable_target}
        TEST_NAME ${ctest_name}
        SOURCE_FILE "${orig_test_src}"
        ERROR_REGEX_LABEL "expected-error"
        ERROR_NUMBER_TARGET_NAME_REGEX "\\.err_([0-9]+)"
      )
    else()
      # Create metatargets for non-fail tests, excluding variant labels:
      cccl_ensure_metatargets(
        ${test_executable_target}
        METATARGET_PATH ${test_basename}
      )

      cmake_language(
        CALL ${self_FUNC_CUSTOMIZE_TEST_COMMAND}
        test_command
        "${orig_test_src}"
        ${test_name}
        ${test_executable_target}
        ${self_EXTRA_ARGN}
      )
      add_test(NAME ${ctest_name} COMMAND ${test_command})
    endif()

    cmake_language(
      CALL ${self_FUNC_CUSTOMIZE_TEST_PROPERTIES}
      "${orig_test_src}"
      ${test_name}
      ${test_executable_target}
      ${ctest_name}
      ${self_EXTRA_ARGN}
    )

    list(APPEND test_names ${test_name})
  endforeach() # Variant

  set(${test_names_var} "${test_names}" PARENT_SCOPE)
endfunction()

#
# Default implementations of cccl_add_tests_from_src customization points:
#

function(_cccl_default_is_src_supported out_var test_src)
  set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(_cccl_default_customize_src out_var test_src)
  set(${out_var} "${test_src}" PARENT_SCOPE)
endfunction()

function(_cccl_default_src_to_basename out_var test_src)
  get_filename_component(basename "${test_src}" NAME_WLE)
  set(${out_var} "${basename}" PARENT_SCOPE)
endfunction()

function(_cccl_default_is_test_supported out_var test_src test_name)
  set(${out_var} TRUE PARENT_SCOPE)
endfunction()

function(_cccl_default_is_compile_fail_test out_var test_src test_name)
  set(${out_var} FALSE PARENT_SCOPE)
  if (test_name MATCHES "[_.]fail(\\.|$)")
    set(${out_var} TRUE PARENT_SCOPE)
  endif()
endfunction()

function(
  _cccl_default_customize_executable
  test_src
  test_name
  test_executable_target
)
  target_link_libraries(
    ${test_executable_target}
    PRIVATE cccl.compiler_interface
  )
endfunction()

function(
  _cccl_default_customize_test_command
  out_var
  test_src
  test_name
  test_executable_target
)
  set(${out_var} "$<TARGET_FILE:${test_executable_target}>" PARENT_SCOPE)
endfunction()

function(
  _cccl_default_customize_test_properties
  test_src
  test_name
  test_executable_target
  ctest_name
)
  # Always skip tests that print CCCL_SKIP_TEST:
  set_tests_properties(
    ${ctest_name}
    PROPERTIES SKIP_REGULAR_EXPRESSION "CCCL_SKIP_TEST"
  )
endfunction()
