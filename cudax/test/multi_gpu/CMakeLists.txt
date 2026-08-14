#===----------------------------------------------------------------------===##
#
# Part of CUDA Experimental in CUDA C++ Core Libraries,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
#===----------------------------------------------------------------------===##

cccl_get_nccl(QUIET)

option(
  cudax_ENABLE_NCCL
  "Enable cudax tests that link against NCCL."
  "${NCCL_FOUND}"
)

# TODO(jfaibussowit):
#
# Remove this once NCCL supports windows and we have added NCCL to the NVHPC
# devcontainers.
macro(hack_cudax_handle_missing_nccl)
  if (NOT cudax_ENABLE_NCCL)
    message(
      AUTHOR_WARNING
      "Function should only be called if we have requested NCCL, this is a developer error"
    )
  endif()

  option(
    HACK_cudax_ALLOW_MISSING_NCCL
    "Hack to work around CI issues since NCCL does not (yet) officially support Windows"
    TRUE
  )

  if (NCCL_FOUND)
    # If we have somehow already found NCCL (e.g. they have shipped support but we have
    # forgotten to remove this function), then we have nothing to do. We still call
    # cccl_get_nccl() though because we want the found version to appear in the logs. The
    # previous call was with QUIET which won't do that.
    cccl_get_nccl(REQUIRED)
  elseif (HACK_cudax_ALLOW_MISSING_NCCL)
    set(cudax_ENABLE_NCCL FALSE)
    set(cudax_ENABLE_NCCL FALSE CACHE BOOL "" FORCE)
  else()
    # This will cause a hard error
    cccl_get_nccl(REQUIRED)
  endif()
endmacro()

if (cudax_ENABLE_NCCL)
  if (WIN32 OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "NVHPC"))
    hack_cudax_handle_missing_nccl()
  else()
    cccl_get_nccl(REQUIRED)
  endif()
endif()

function(cudax_add_multi_gpu_test sub_prefix target_name_var source)
  cmake_path(GET source STEM filename)
  cudax_add_catch2_test("${target_name_var}" "multi_gpu.${sub_prefix}.${filename}" "${source}" ${ARGN})
  target_include_directories(
    "${${target_name_var}}"
    PRIVATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/include"
  )
  set("${target_name_var}" "${${target_name_var}}" PARENT_SCOPE)
endfunction()

#[=======================================================================[.rst:
cudax_split_multi_gpu_tests
---------------------------

Split a list of test sources into the ones that are expected to compile and the ones that
are expected to fail to compile.

A source is expected to fail to compile if its file name ends in ``_fail``. The two kinds
need different registration, because a fail test is not a Catch2 test and must not be built
as part of ``all``. So pass the first result to :command:`cudax_add_multi_gpu_test` and the
second to :command:`cudax_add_multi_gpu_fail_test`.

Arguments
^^^^^^^^^

``srcs``
  The list of test sources to split. Pass the list variable by name, not by value.

``pass_srcs_var``
  The variable in which to store the sources that are expected to compile, in the parent
  scope.

``fail_srcs_var``
  The variable in which to store the sources that are expected to fail to compile, in the
  parent scope.

#]=======================================================================]
function(cudax_split_multi_gpu_tests srcs pass_srcs_var fail_srcs_var)
  set(pass_srcs)
  set(fail_srcs)

  foreach (src IN LISTS ${srcs})
    cmake_path(GET src STEM filename)
    if ("${filename}" MATCHES "_fail$")
      list(APPEND fail_srcs "${src}")
    else()
      list(APPEND pass_srcs "${src}")
    endif()
  endforeach()

  set("${pass_srcs_var}" "${pass_srcs}" PARENT_SCOPE)
  set("${fail_srcs_var}" "${fail_srcs}" PARENT_SCOPE)
endfunction()

#[=======================================================================[.rst:
cudax_add_multi_gpu_fail_test
-----------------------------

Add a multi-GPU test that is expected to fail to compile, and register it with ctest. Use
this for sources that follow the ``_fail.cu`` naming convention.

The created target is excluded from ``all``, because building it is the failure the test
looks for. The test itself builds the target and passes only if the compiler output matches
the expected error. So the source must declare the error it expects with a comment of the
form:

.. code-block:: c++

  // expected-error {{"the diagnostic text to match"}}

The target is never linked or run, so it needs no Catch2 main.

The created target is named ``cudax.test.multi_gpu.<sub_prefix>.error.<source stem>``.

Arguments
^^^^^^^^^

``sub_prefix``
  The dotted name of the group the test belongs to, for example ``algorithms.reduce``. Use
  the same value as for the corresponding :command:`cudax_add_multi_gpu_test` calls.

``target_name_var``
  The variable in which to store the created target name, in the parent scope. Useful for
  modifying the target after creation, for example to link an extra library.

``source``
  The source file that implements the test.

#]=======================================================================]
function(cudax_add_multi_gpu_fail_test sub_prefix target_name_var source)
  cmake_path(GET source STEM filename)

  set(test_target "cudax.test.multi_gpu.${sub_prefix}.error.${filename}")

  # The test never links or runs, so it needs no Catch2 main. NO_METATARGETS keeps it out of
  # the aggregate build targets, since it is expected to fail to compile.
  cccl_add_executable(
    ${test_target}
    SOURCES "${source}"
    NO_METATARGETS
    NO_CLANG_TIDY
  )
  target_include_directories(
    ${test_target}
    PRIVATE "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/include"
  )
  target_link_libraries(
    ${test_target}
    PRIVATE #
      cudax.compiler_interface
      CUDA::cudart_static
  )
  cccl_add_xfail_compile_target_test(
    ${test_target}
    SOURCE_FILE "${source}"
    ERROR_REGEX_LABEL "expected-error"
  )

  set("${target_name_var}" "${test_target}" PARENT_SCOPE)
endfunction()

add_subdirectory(algorithms)
add_subdirectory(communicators)
add_subdirectory(concepts)
add_subdirectory(nccl)
