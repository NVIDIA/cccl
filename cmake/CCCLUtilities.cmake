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


# Add a build-and-test CTest.
# - full_test_name_var will be set to the full name of the test.
# - name_prefix is the prefix of the test's name (e.g. `cccl.test.cmake`)
# - subdir is the relative path to the test project directory.
# - test_id is used to generate a unique name for this test, allowing the
#   subdir to be reused.
# - Any additional args will be passed to the project configure step.
function(cccl_add_compile_test full_test_name_var name_prefix subdir test_id)
  set(test_name ${name_prefix}.${subdir}.${test_id})
  set(src_dir "${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${subdir}/${test_id}")
  add_test(NAME ${test_name}
    COMMAND "${CMAKE_CTEST_COMMAND}"
      --build-and-test "${src_dir}" "${build_dir}"
      --build-generator "${CMAKE_GENERATOR}"
      --build-options
        ${ARGN}
      --test-command "${CMAKE_CTEST_COMMAND}" --output-on-failure
  )
  set(${full_test_name_var} ${test_name} PARENT_SCOPE)
endfunction()
