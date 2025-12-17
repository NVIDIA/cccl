# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Passes all args directly to execute_process while setting up the following
# results variables and propagating them to the caller's scope:
#
# - cccl_process_exit_code
# - cccl_process_stdout
# - cccl_process_stderr
#
# If the command
# is not successful (e.g. the last command does not return zero), a non-fatal
# warning is printed.
function(cccl_execute_non_fatal_process)
  # Skip parsing this function's signature -- it is handled by .gersemi/ext/cccl.py.
  # gersemi: ignore

  execute_process(
    ${ARGN}
    RESULT_VARIABLE cccl_process_exit_code
    OUTPUT_VARIABLE cccl_process_stdout
    ERROR_VARIABLE cccl_process_stderr
  )

  if (NOT cccl_process_exit_code EQUAL 0)
    message(
      WARNING
      "execute_process failed with non-zero exit code: ${cccl_process_exit_code}\n"
      "${ARGN}\n"
      "stdout:\n${cccl_process_stdout}\n"
      "stderr:\n${cccl_process_stderr}\n"
    )
  endif()

  set(cccl_process_exit_code "${cccl_process_exit_code}" PARENT_SCOPE)
  set(cccl_process_stdout "${cccl_process_stdout}" PARENT_SCOPE)
  set(cccl_process_stderr "${cccl_process_stderr}" PARENT_SCOPE)
endfunction()

# Add a build-and-test CTest.
# - full_test_name_var will be set to the full name of the test.
# - name_prefix is the prefix of the test's name (e.g. `cccl.test.cmake`)
# - subdir is the relative path to the test project directory.
# - test_id is used to generate a unique name for this test, allowing the
#   subdir to be reused.
# - CTEST_COMMAND is the command to use for running CTest [optional]
# - Any additional args will be passed to the project configure step.
function(cccl_add_compile_test full_test_name_var name_prefix subdir test_id)
  set(options)
  set(oneValueArgs CTEST_COMMAND)
  set(multiValueArgs)
  cmake_parse_arguments(
    cccl_compile_test
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  if (NOT DEFINED cccl_compile_test_CTEST_COMMAND)
    set(cccl_compile_test_CTEST_COMMAND "${CMAKE_CTEST_COMMAND}")
  endif()

  set(test_name ${name_prefix}.${subdir}.${test_id})
  set(src_dir "${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${subdir}/${test_id}")
  add_test(
    NAME ${test_name}
    # gersemi: off
    COMMAND
      "${cccl_compile_test_CTEST_COMMAND}"
        --build-and-test "${src_dir}" "${build_dir}"
        --build-generator "${CMAKE_GENERATOR}"
        --build-options ${cccl_compile_test_UNPARSED_ARGUMENTS}
        --test-command "${cccl_compile_test_CTEST_COMMAND}" --output-on-failure
    # gersemi: on
  )
  set(${full_test_name_var} ${test_name} PARENT_SCOPE)
endfunction()

# cccl_add_xfail_compile_target_test(
#   <target_name>
#   [TEST_NAME <test_name>]
#   [ERROR_REGEX <regex>]
#   [SOURCE_FILE <source_file>]
#   [ERROR_REGEX_LABEL <error_string>]
#   [ERROR_NUMBER <error_number>]
#   [ERROR_NUMBER_TARGET_NAME_REGEX <regex>]
# )
#
# Given a configured build target that is expected to fail to compile:
# - Mark the target as excluded from the `all` target.
# - Create a CTest test that compiles the target. If TEST_NAME is provided, it is used.
#   Otherwise, the target_name is used as the test name.
# - When the test runs, it passes if exactly one of the following conditions is met:
#   - A provided / detected error regex matches the compilation output, ignoring exit code.
#   - No error regex is provided / detected, and the compilation fails.
#
# An error regex may be explicitly provided via ERROR_REGEX, or it may be
# detected by scanning the SOURCE_FILE for a specially formatted comment.
#
# If ERROR_REGEX_LABEL is provided, the SOURCE_FILE will read, looking for a comment of the form:
#
# // <ERROR_REGEX_LABEL> {{"error_regex"}}
#
# An error number may be appended to the ERROR_REGEX_LABEL in the comment:
#
# // <ERROR_REGEX_LABEL>-<error_number> {{"error_regex"}}
#
# If ERROR_NUMBER_TARGET_NAME_REGEX is specified, the regex is used to capture
# the error_number from the target name. If target_name is
# "cccl.test.my_test.err_5.foo_3" and ERROR_NUMBER_TARGET_NAME_REGEX is
# "\\.err_([0-9]+)", the captured error number "5."
#
# // <ERROR_REGEX_LABEL>-<captured_error_number> {{"error_regex"}}
#
# If ERROR_NUMBER is provided, ERROR_NUMBER_TARGET_NAME_REGEX is ignored.
# If ERROR_NUMBER_TARGET_NAME_REGEX is provided but does not match, a plain ERROR_REGEX_LABEL is used.
#
# If both SOURCE_FILE and ERROR_REGEX_LABEL are provided, the source file will be added to the
# current directory's CMAKE_CONFIGURE_DEPENDS to ensure that changes to the file will re-trigger CMake.
function(cccl_add_xfail_compile_target_test target_name)
  set(options)
  set(
    oneValueArgs
    TEST_NAME
    ERROR_REGEX
    SOURCE_FILE
    ERROR_REGEX_LABEL
    ERROR_NUMBER
    ERROR_NUMBER_TARGET_NAME_REGEX
  )
  set(multiValueArgs)
  cmake_parse_arguments(
    cccl_xfail
    "${options}"
    "${oneValueArgs}"
    "${multiValueArgs}"
    ${ARGN}
  )

  if (cccl_xfail_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unparsed arguments: ${cccl_xfail_UNPARSED_ARGUMENTS}")
  endif()

  set(test_name "${target_name}")
  if (DEFINED cccl_xfail_TEST_NAME)
    set(test_name "${cccl_xfail_TEST_NAME}")
  endif()

  set(regex)
  if (DEFINED cccl_xfail_ERROR_REGEX)
    set(regex "${cccl_xfail_ERROR_REGEX}")
  elseif (
    DEFINED cccl_xfail_SOURCE_FILE
    AND DEFINED cccl_xfail_ERROR_REGEX_LABEL
  )
    get_filename_component(src_absolute "${cccl_xfail_SOURCE_FILE}" ABSOLUTE)
    set(error_label_regex "${cccl_xfail_ERROR_REGEX_LABEL}")

    # Cache all error label matches (with and without error numbers) as global properties.
    # This avoids re-reading and re-parsing the source file multiple times if multiple
    # tests are added for the same source file. Properties are used instead of cache variables
    # to ensure that the source is not cached in between CMake executions.
    string(MD5 source_filename_md5 "${src_absolute}")
    set(error_cache_property "_cccl_xfail_error_cache_${source_filename_md5}")
    get_property(error_cache_set GLOBAL PROPERTY "${error_cache_property}" SET)
    if (error_cache_set)
      get_property(error_cache GLOBAL PROPERTY "${error_cache_property}")
    else()
      file(READ "${src_absolute}" source_contents)
      string(
        REGEX MATCHALL
        "//[ \t]*${error_label_regex}(-[0-9]+)?[ \t]*{{\"([^\"]+)\"}}"
        error_cache
        "${source_contents}"
      )
      set_property(GLOBAL PROPERTY "${error_cache_property}" "${error_cache}")
    endif()

    # Changes to the source file should re-run CMake to pick-up new error specs:
    set_property(
      DIRECTORY
      APPEND
      PROPERTY CMAKE_CONFIGURE_DEPENDS "${src_absolute}"
    )

    set(error_number)
    if (DEFINED cccl_xfail_ERROR_NUMBER)
      set(error_number "${cccl_xfail_ERROR_NUMBER}")
    elseif (DEFINED cccl_xfail_ERROR_NUMBER_TARGET_NAME_REGEX)
      string(
        REGEX MATCH
        "${cccl_xfail_ERROR_NUMBER_TARGET_NAME_REGEX}"
        matched
        ${target_name}
      )
      if (matched)
        set(error_number "${CMAKE_MATCH_1}")
      endif()
    endif()

    # Look for a labeled error with the specific error number.
    if (NOT "${error_number}" STREQUAL "") # Check strings to allow "0"
      string(
        REGEX MATCH
        "//[ \t]*${error_label_regex}-${error_number}[ \t]*{{\"([^\"]+)\"}}"
        matched
        "${error_cache}"
      )
      if (matched)
        set(regex "${CMAKE_MATCH_1}")
      endif()
    endif()

    if (NOT regex)
      # Look for a labeled error without an error number.
      string(
        REGEX MATCH
        "//[ \t]*${error_label_regex}[ \t]*{{\"([^\"]+)\"}}"
        matched
        "${error_cache}"
      )
      if (matched)
        set(regex "${CMAKE_MATCH_1}")
      endif()
    endif()
  endif()

  message(VERBOSE "CCCL: Adding XFAIL test: ${test_name}")
  if (regex)
    message(VERBOSE "CCCL:   with expected regex: '${regex}'")
  endif()

  set_target_properties(${test_target} PROPERTIES EXCLUDE_FROM_ALL true)

  # The same target may be reused for multiple tests, and the output file
  # may exist if using a regex to check for warnings. Add a setup fixture to
  # delete the output file before each test run.
  if (NOT TEST ${target_name}.clean)
    add_test(
      NAME ${target_name}.clean
      # gersemi: off
      COMMAND
        "${CMAKE_COMMAND}" -E rm -f
          "$<TARGET_FILE:${target_name}>"
          "$<TARGET_OBJECTS:${target_name}>"
      # gersemi: on
    )
    set_tests_properties(
      ${test_name}.clean
      PROPERTIES FIXTURES_SETUP ${target_name}.clean
    )
  endif()

  add_test(
    NAME ${test_name}
    # gersemi: off
    COMMAND
      "${CMAKE_COMMAND}"
        --build "${CMAKE_BINARY_DIR}"
        --target ${test_target}
        --config $<CONFIGURATION>
    # gersemi: on
  )
  set_tests_properties(
    ${test_name}
    PROPERTIES FIXTURES_CLEANUP ${target_name}.clean
  )

  if (regex)
    set_tests_properties(
      ${test_name}
      PROPERTIES PASS_REGULAR_EXPRESSION "${regex}"
    )
  else()
    set_tests_properties(${test_name} PROPERTIES WILL_FAIL true)
  endif()
endfunction()
