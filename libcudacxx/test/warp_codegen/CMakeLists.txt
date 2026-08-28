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

libcudacxx_codegen_check_tools(libcudacxx_codegen_tests_enabled)
if (NOT libcudacxx_codegen_tests_enabled)
  return()
endif()

libcudacxx_codegen_get_cuda_architectures(
  requested_warp_codegen_cuda_archs
  SASS
)
set(warp_codegen_cuda_archs)
foreach (arch IN LISTS requested_warp_codegen_cuda_archs)
  string(REGEX MATCH "^[0-9]+" numeric_arch "${arch}")
  if (numeric_arch GREATER_EQUAL 90)
    list(APPEND warp_codegen_cuda_archs "${arch}")
  endif()
endforeach()
if (NOT warp_codegen_cuda_archs)
  message(
    STATUS
    "No compatible CUDA architectures requested; skipping warp SASS codegen tests"
  )
  return()
endif()

file(GLOB warp_codegen_tests CONFIGURE_DEPENDS "*.cu")

set(
  warp_codegen_stress_test
  "${CMAKE_CURRENT_SOURCE_DIR}/warp_shuffle_stress.cu"
)
list(REMOVE_ITEM warp_codegen_tests "${warp_codegen_stress_test}")

libcudacxx_codegen_add_sass_tests(
  AGGREGATE_TARGET libcudacxx.test.warp.sass
  TARGET_PREFIX warp_codegen
  ARCHITECTURES ${warp_codegen_cuda_archs}
  TESTS ${warp_codegen_tests}
)

foreach (arch IN LISTS warp_codegen_cuda_archs)
  set(target_name "warp_codegen_sass_sm${arch}_warp_shuffle_stress")
  set(
    check_target_name
    "libcudacxx.test.warp.sass.sm${arch}.warp_shuffle_stress.check"
  )

  libcudacxx_codegen_add_test(
    SEPARABLE_COMPILATION
    AGGREGATE_TARGET libcudacxx.test.warp.sass
    TARGET_PREFIX warp_codegen
    CODE_KIND sass
    ARCH "${arch}"
    TEST "${warp_codegen_stress_test}"
  )
  add_custom_target(
    ${check_target_name}
    DEPENDS ${target_name}
    COMMAND
      "${Python_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/compare_register_usage.py"
      "--cuobjdump=${libcudacxx_codegen_cuobjdump}"
      "--binary=$<TARGET_FILE:${target_name}>"
  )
  cccl_ensure_metatargets(${check_target_name})
endforeach()
