##===----------------------------------------------------------------------===##
##
## Part of nvrtcc in CUDA Core Compute Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

# 1. Add -use-nvrtc test.
add_executable(test.nvrtcc_options.use_nvrtc.false use_nvrtc.false.cu)
add_dependencies(nvrtcc.test test.nvrtcc_options.use_nvrtc.false)
add_test(
  NAME test.nvrtcc_options.use_nvrtc.false.run
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test.nvrtcc_options.use_nvrtc.false
)

add_executable(test.nvrtcc_options.use_nvrtc use_nvrtc.true.cu)
add_dependencies(nvrtcc.test test.nvrtcc_options.use_nvrtc)
target_compile_options(test.nvrtcc_options.use_nvrtc PRIVATE "--use-nvrtc")
add_test(
  NAME test.nvrtcc_options.use_nvrtc.run
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test.nvrtcc_options.use_nvrtc
)

add_executable(test.nvrtcc_options.use_nvrtc.short use_nvrtc.true.cu)
add_dependencies(nvrtcc.test test.nvrtcc_options.use_nvrtc.short)
target_compile_options(test.nvrtcc_options.use_nvrtc.short PRIVATE "-use-nvrtc")
add_test(
  NAME test.nvrtcc_options.use_nvrtc.short.run
  COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test.nvrtcc_options.use_nvrtc.short
)

# 2. Add -Xnvrtc test. TODO: make --nvrtc-options -my_opt work.
add_compile_only_test(test.nvrtcc_options.nvrtc_options nvrtc_options.cu)
target_compile_options(
  test.nvrtcc_options.nvrtc_options
  PRIVATE
    "--nvrtc-options=-builtin-move-forward=false"
    "--nvrtc-options=-builtin-initializer-list=false"
    "--nvrtc-options=-DNVRTC_ONLY_MACRO=42"
)

add_compile_only_test(test.nvrtcc_options.nvrtc_options.short nvrtc_options.cu)
target_compile_options(
  test.nvrtcc_options.nvrtc_options.short
  PRIVATE
    "-Xnvrtc=-builtin-move-forward=false"
    "-Xnvrtc=-builtin-initializer-list=false"
    "-Xnvrtc=-DNVRTC_ONLY_MACRO=42"
)
