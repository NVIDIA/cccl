##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

add_custom_target(libcudacxx.test.simd.ptx)
add_custom_target(libcudacxx.test.simd.sass)

#-----------------------------------------------------------------------------------------------------------------------
# SETUP: skip unsupported compilers, find tools, set up CUDA architectures

# GCC7 and NVCC 12.0 do not vectorize load/store.
if (
  CMAKE_CXX_COMPILER_ID STREQUAL GNU
  AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7
  AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8
)
  message("-- Skipping simd codegen tests with GCC 7 host compiler")
  return()
endif()

if (
  CMAKE_CUDA_COMPILER_ID STREQUAL NVIDIA
  AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0
  AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.1
)
  message("-- Skipping simd codegen tests with NVCC 12.0")
  return()
endif()

# skip nvc++
if (CMAKE_CXX_COMPILER_ID STREQUAL NVHPC)
  message("-- Skipping simd codegen tests with NVHPC host compiler")
  return()
endif()

find_program(
  filecheck
  NAMES FileCheck FileCheck-21 FileCheck-20 FileCheck-19 FileCheck-18
)

if (filecheck)
  message("-- ${filecheck} found... building simd codegen tests")
else()
  return()
endif()

find_program(cuobjdump "cuobjdump" REQUIRED)
find_program(bash "bash" REQUIRED)

set(
  dump_and_check_script
  "${CMAKE_CURRENT_SOURCE_DIR}/../atomic_codegen/dump_and_check.bash"
)

set(simd_codegen_sass_cuda_archs 80 90)
if (
  CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8
  AND NOT CMAKE_CUDA_COMPILER_ID STREQUAL Clang
)
  list(APPEND simd_codegen_sass_cuda_archs 100 120)
endif()

#-----------------------------------------------------------------------------------------------------------------------
# HELPER FUNCTIONS

function(simd_codegen_set_cuda_arch target_name arch)
  set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES "${arch}")
endfunction()

# Given the test file content, extract the SM archs that are checked for.
# SMXX:       any arch
# SM100-PLUS: SM100 and newer
function(simd_codegen_get_sass_check_prefixes out_var test_contents arch)
  set(check_prefixes SMXX)
  set(arch_prefix "SM${arch}")

  string(
    REGEX MATCH
    "; ${arch_prefix}(:|-LABEL:|-NOT:|-NEXT:|-SAME:|-DAG:|-COUNT:|-EMPTY:)"
    has_arch_prefix
    "${test_contents}"
  )
  if (has_arch_prefix)
    list(APPEND check_prefixes "${arch_prefix}")
  endif()

  string(
    REGEX MATCHALL
    "; SM[0-9]+-PLUS(:|-[A-Z]+:)"
    plus_prefixes
    "${test_contents}"
  )
  foreach (plus_prefix IN LISTS plus_prefixes)
    string(REGEX REPLACE ".*SM([0-9]+)-PLUS.*" "\\1" plus_arch "${plus_prefix}")
    if (arch GREATER_EQUAL plus_arch)
      list(APPEND check_prefixes "SM${plus_arch}-PLUS")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES check_prefixes)
  list(JOIN check_prefixes "," check_prefixes)

  set(${out_var} "${check_prefixes}" PARENT_SCOPE)
endfunction()

# configure the library target for the test
function(simd_codegen_add_library_target out_var kind arch test_path)
  cmake_path(GET test_path FILENAME test_file)
  cmake_path(REMOVE_EXTENSION test_file LAST_ONLY OUTPUT_VARIABLE test_name)

  set(target_name "simd_codegen_${kind}_sm${arch}_${test_name}")

  add_library(${target_name} STATIC "${test_path}")

  simd_codegen_set_cuda_arch(${target_name} "${arch}")
  target_compile_options(${target_name} PRIVATE "-Wno-comment")

  target_include_directories(
    ${target_name}
    PRIVATE "${libcudacxx_SOURCE_DIR}/include"
  )
  set(${out_var} "${target_name}" PARENT_SCOPE)
endfunction()

# generate the check target for the test (FILECHECK)
function(
  simd_codegen_add_check_target
  aggregate_target
  target_name
  test_path
  check_prefixes
)
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" check_suffix "${check_prefixes}")
  set(check_target_name "${target_name}_${check_suffix}_check")

  add_custom_target(
    ${check_target_name}
    DEPENDS ${target_name}
    # gersemi: off
    COMMAND
      ${CMAKE_COMMAND} -E env "FILECHECK=${filecheck}"
        "${dump_and_check_script}"
        $<TARGET_FILE:${target_name}>
        "${test_path}"
        "${check_prefixes}"
        ${ARGN}
    # gersemi: on
  )
  add_dependencies(${aggregate_target} ${check_target_name})
endfunction()

# generate the command to run the PTX tests
function(simd_codegen_add_ptx_tests arch prefix_list test_list)
  foreach (test_path IN LISTS ${test_list})
    simd_codegen_add_library_target(target_name ptx "${arch}" "${test_path}")

    # Clang stopped emitting PTX in clang20. Add flags to re-enable it.
    if (
      CMAKE_CUDA_COMPILER_ID STREQUAL Clang
      AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 20
    )
      target_compile_options(
        ${target_name}
        PRIVATE "--cuda-include-ptx=sm_${arch}"
      )
    endif()

    foreach (prefix IN LISTS ${prefix_list})
      simd_codegen_add_check_target(libcudacxx.test.simd.ptx ${target_name} "${test_path}" "${prefix}")
    endforeach()
  endforeach()
endfunction()

# generate the command to run the SASS tests
function(simd_codegen_add_sass_test test_path)
  file(READ "${test_path}" test_contents)

  set(test_archs)
  foreach (arch IN LISTS simd_codegen_sass_cuda_archs)
    simd_codegen_get_sass_check_prefixes(check_prefixes "${test_contents}" "${arch}")
    if (NOT "${check_prefixes}" STREQUAL "SMXX")
      list(APPEND test_archs "${arch}")
    endif()
  endforeach()
  if (NOT test_archs)
    set(test_archs ${simd_codegen_sass_cuda_archs})
  endif()

  foreach (arch IN LISTS test_archs)
    simd_codegen_add_library_target(target_name sass "${arch}" "${test_path}")
    simd_codegen_get_sass_check_prefixes(check_prefixes "${test_contents}" "${arch}")

    string(FIND "${test_contents}" "__device__" has_device_function)
    if (NOT has_device_function EQUAL -1)
      target_compile_options(${target_name} PRIVATE "-dc")
    endif()

    simd_codegen_add_check_target(libcudacxx.test.simd.sass ${target_name} "${test_path}" "${check_prefixes}" --dump-sass)
  endforeach()
endfunction()

set(simd_codegen_sass_tests)
file(GLOB simd_codegen_sass_tests "*.cu")

add_subdirectory(load_store)

foreach (test_path IN LISTS simd_codegen_sass_tests)
  simd_codegen_add_sass_test("${test_path}")
endforeach()
