#===----------------------------------------------------------------------===##
#
# Part of CUDA Experimental in CUDA C++ Core Libraries,
# under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#
#===----------------------------------------------------------------------===##

# Also compile a complete NCCL-communicator sort without linking libnccl. With
# NCCL disabled this covers header absence; with it enabled it covers optional
# device SDK entry points without masking hard references through NCCL::nccl.
cudax_add_multi_gpu_test("algorithms.sort" link_target "${CMAKE_CURRENT_SOURCE_DIR}/optional_link.cu")
if (cudax_ENABLE_NCCL)
  target_include_directories(
    ${link_target}
    SYSTEM
    PRIVATE $<TARGET_PROPERTY:NCCL::nccl,INTERFACE_INCLUDE_DIRECTORIES>
  )
else()
  return()
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Model an installed runtime that either lacks device symbols, or is older
  # than the device SDK. Neither test requires a GPU communication allocation.
  foreach (case IN ITEMS missing_symbols older_runtime)
    set(stub_target "${link_target}.${case}.library")
    add_library(${stub_target} SHARED linkage/library.cpp)
    target_include_directories(
      ${stub_target}
      SYSTEM
      PRIVATE
        $<TARGET_PROPERTY:NCCL::nccl,INTERFACE_INCLUDE_DIRECTORIES>
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    set_target_properties(
      ${stub_target}
      PROPERTIES
        OUTPUT_NAME nccl
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${case}"
    )
    if (case STREQUAL "older_runtime")
      target_compile_definitions(${stub_target} PRIVATE CUDAX_TEST_OLDER_NCCL=1)
    endif()
    add_dependencies(${link_target} ${stub_target})
    add_test(
      NAME "${link_target}.${case}"
      COMMAND
        sh -c
        "LD_LIBRARY_PATH=\"$1:\${LD_LIBRARY_PATH:-}\" CUDAX_TEST_EXPECT_NO_DEVICE_API=1 exec \"$2\""
        sh "$<TARGET_FILE_DIR:${stub_target}>" "$<TARGET_FILE:${link_target}>"
    )
    set_tests_properties("${link_target}.${case}" PROPERTIES SKIP_RETURN_CODE 4)
  endforeach()
endif()

file(GLOB test_srcs LIST_DIRECTORIES FALSE CONFIGURE_DEPENDS *.cu *.cpp)

foreach (src IN LISTS test_srcs)
  if (src STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/optional_link.cu")
    continue()
  endif()
  cudax_add_multi_gpu_test("algorithms.sort" test_target "${src}")
  target_link_libraries(${test_target} PRIVATE NCCL::nccl)
  # Catch2 returns 4 when every case is skipped (e.g. one GPU or an older
  # device SDK). Unsupported configurations are skips, not failed assertions.
  set_tests_properties(${test_target} PROPERTIES SKIP_RETURN_CODE 4)
endforeach()
