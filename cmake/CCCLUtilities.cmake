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

# Enable or disable relocatable device code (RDC) for a given target.
function(cccl_set_rdc target_name rdc_enabled)
  if (rdc_enabled)
    set_target_properties(
      ${target_name}
      PROPERTIES #
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )
  else()
    set_target_properties(
      ${target_name}
      PROPERTIES CUDA_SEPARABLE_COMPILATION OFF
    )
  endif()
endfunction()

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
