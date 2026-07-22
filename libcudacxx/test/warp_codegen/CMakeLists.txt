##===----------------------------------------------------------------------===##
##
## Part of libcu++ in the CUDA C++ Core Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

add_custom_target(libcudacxx.test.warp.sass)

if (
  CMAKE_CXX_COMPILER_ID STREQUAL NVHPC
  OR CMAKE_CUDA_COMPILER_ID STREQUAL Clang
)
  message("-- Skipping warp codegen tests with ${CMAKE_CUDA_COMPILER_ID}")
  return()
endif()

find_program(cuobjdump "cuobjdump" REQUIRED)
find_program(
  filecheck
  NAMES FileCheck FileCheck-21 FileCheck-20 FileCheck-19 FileCheck-18
)
if (NOT filecheck)
  return()
endif()

set(
  dump_and_check_script
  "${CMAKE_CURRENT_SOURCE_DIR}/../atomic_codegen/dump_and_check.bash"
)
message("-- Building warp codegen tests")

set(warp_codegen_cuda_archs 90)
if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  list(APPEND warp_codegen_cuda_archs 100 120)
endif()

file(GLOB warp_codegen_tests "*.cu")

foreach (test_path IN LISTS warp_codegen_tests)
  cmake_path(GET test_path FILENAME test_file)
  cmake_path(REMOVE_EXTENSION test_file LAST_ONLY OUTPUT_VARIABLE test_name)

  foreach (arch IN LISTS warp_codegen_cuda_archs)
    set(target_name "warp_codegen_sm${arch}_${test_name}")
    set(check_target_name "${target_name}_check")

    add_library(${target_name} STATIC "${test_path}")
    set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES OFF)
    target_compile_options(
      ${target_name}
      PRIVATE
        "-dc"
        "-Wno-comment"
        "--generate-code=arch=compute_${arch},code=sm_${arch}"
    )
    target_include_directories(
      ${target_name}
      PRIVATE "${libcudacxx_SOURCE_DIR}/include"
    )

    if (test_name STREQUAL "warp_shuffle_stress")
      add_custom_target(
        ${check_target_name}
        DEPENDS ${target_name}
        COMMAND
          "${Python_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/compare_register_usage.py"
          "--cuobjdump=${cuobjdump}" "--binary=$<TARGET_FILE:${target_name}>"
      )
    else()
      add_custom_target(
        ${check_target_name}
        DEPENDS ${target_name}
        COMMAND
          ${CMAKE_COMMAND} -E env "FILECHECK=${filecheck}"
          "${dump_and_check_script}" $<TARGET_FILE:${target_name}>
          "${test_path}" "SMXX" --dump-sass
      )
    endif()
    add_dependencies(libcudacxx.test.warp.sass ${check_target_name})
  endforeach()
endforeach()
