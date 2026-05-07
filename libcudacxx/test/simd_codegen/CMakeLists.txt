##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

add_custom_target(libcudacxx.test.simd.sass)

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

set(libcudacxx_simd_codegen_tests)
if (NOT "NVHPC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
  file(GLOB libcudacxx_simd_codegen_tests "*.cu")
endif()

set(simd_codegen_cuda_archs 80 90)
if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  list(APPEND simd_codegen_cuda_archs 100 120)
endif()

function(simd_codegen_add_tests test_path)
  cmake_path(GET test_path FILENAME test_file)
  cmake_path(REMOVE_EXTENSION test_file LAST_ONLY OUTPUT_VARIABLE test_name)

  file(READ "${test_path}" test_contents)
  string(
    REGEX MATCH
    "SM[0-9][0-9]*"
    has_arch_specific_prefix
    "${test_contents}"
  )

  # search for SMXX prefixes in the test contents
  set(test_archs)
  if (has_arch_specific_prefix)
    foreach (arch IN LISTS simd_codegen_cuda_archs)
      string(FIND "${test_contents}" "SM${arch}" arch_specific_prefix)
      if (NOT arch_specific_prefix EQUAL -1)
        list(APPEND test_archs "${arch}")
      endif()
    endforeach()
  else()
    set(test_archs ${simd_codegen_cuda_archs})
  endif()

  # Run tests for each architecture specified in the test file
  foreach (arch IN LISTS test_archs)
    set(target_name "simd_codegen_sm${arch}_${test_name}")
    set(check_prefixes "SMXX")
    if (has_arch_specific_prefix)
      string(APPEND check_prefixes ",SM${arch}")
    endif()

    add_library(${target_name} STATIC "${test_path}")

    set_target_properties(
      ${target_name}
      PROPERTIES CUDA_ARCHITECTURES "${arch}"
    )

    target_compile_options(${target_name} PRIVATE "-Wno-comment")

    target_include_directories(
      ${target_name}
      PRIVATE "${libcudacxx_SOURCE_DIR}/include"
    )
    add_dependencies(libcudacxx.test.simd.sass ${target_name})

    add_custom_command(
      TARGET libcudacxx.test.simd.sass
      POST_BUILD
      # gersemi: off
      COMMAND
        ${CMAKE_COMMAND} -E env "FILECHECK=${filecheck}"
          "${CMAKE_CURRENT_SOURCE_DIR}/../atomic_codegen/dump_and_check.bash"
          $<TARGET_FILE:${target_name}>
          "${test_path}"
          "${check_prefixes}"
          --dump-sass
      # gersemi: on
    )
  endforeach()
endfunction()

foreach (test_path IN LISTS libcudacxx_simd_codegen_tests)
  simd_codegen_add_tests("${test_path}")
endforeach()
