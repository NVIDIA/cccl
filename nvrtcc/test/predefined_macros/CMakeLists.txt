##===----------------------------------------------------------------------===##
##
## Part of nvrtcc in CUDA Core Compute Libraries,
## under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
## SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
##
##===----------------------------------------------------------------------===##

# 1. Add default macros test.
add_compile_only_test(test.predefined_macros.default default.cu)

# 2. Add __CUDA_ARCH_LIST__ test.
add_compile_only_test(test.predefined_macros.cuda_arch_list __CUDA_ARCH_LIST__.cu)
set_target_properties(
  test.predefined_macros.cuda_arch_list
  PROPERTIES CUDA_ARCHITECTURES "75;80;89;90"
)

# 3. Add __CUDACC_DEBUG__ test.
add_compile_only_test(test.predefined_macros.cudacc_debug __CUDACC_DEBUG__.cu)
target_compile_options(
  test.predefined_macros.cudacc_debug
  PRIVATE --device-debug
)

add_compile_only_test(test.predefined_macros.cudacc_debug_short __CUDACC_DEBUG__.cu)
target_compile_options(test.predefined_macros.cudacc_debug_short PRIVATE -G)

# 4. Add __CUDACC_EWP__ test.
add_compile_only_test(test.predefined_macros.cudacc_ewp __CUDACC_EWP__.cu)
target_compile_options(
  test.predefined_macros.cudacc_ewp
  PRIVATE --extensible-whole-program
)

add_compile_only_test(test.predefined_macros.cudacc_ewp_short __CUDACC_EWP__.cu)
target_compile_options(test.predefined_macros.cudacc_ewp_short PRIVATE -ewp)

# 5. Add __CUDACC_RDC__ test.
add_compile_only_test(test.predefined_macros.cudacc_rdc __CUDACC_RDC__.cu)
target_compile_options(
  test.predefined_macros.cudacc_rdc
  PRIVATE --relocatable-device-code=true
)

add_compile_only_test(test.predefined_macros.cudacc_rdc_short __CUDACC_RDC__.cu)
target_compile_options(
  test.predefined_macros.cudacc_rdc_short
  PRIVATE -rdc=true
)

add_compile_only_test(test.predefined_macros.cudacc_rdc_alias __CUDACC_RDC__.cu)
target_compile_options(
  test.predefined_macros.cudacc_rdc_alias
  PRIVATE --device-c
)

add_compile_only_test(test.predefined_macros.cudacc_rdc_alias_short __CUDACC_RDC__.cu)
target_compile_options(
  test.predefined_macros.cudacc_rdc_alias_short
  PRIVATE -dc
)

# 6. Add __CUDACC_RTC_FLOAT128__ test.
add_compile_only_test(test.predefined_macros.cudacc_rtc_float128 __CUDACC_RTC_FLOAT128__.cu)
target_compile_options(
  test.predefined_macros.cudacc_rtc_float128
  PRIVATE -Xnvrtc=--device-float128
)
set_target_properties(
  test.predefined_macros.cudacc_rtc_float128
  PROPERTIES CUDA_ARCHITECTURES "90;100"
)

add_compile_only_test(test.predefined_macros.cudacc_rtc_float128_short __CUDACC_RTC_FLOAT128__.cu)
target_compile_options(
  test.predefined_macros.cudacc_rtc_float128_short
  PRIVATE -Xnvrtc=-device-float128
)
set_target_properties(
  test.predefined_macros.cudacc_rtc_float128_short
  PROPERTIES CUDA_ARCHITECTURES "90;100"
)

# 7. Add __CUDACC_RTC_INT128__ test.
add_compile_only_test(test.predefined_macros.cudacc_rtc_int128 __CUDACC_RTC_INT128__.cu)
target_compile_options(
  test.predefined_macros.cudacc_rtc_int128
  PRIVATE -Xnvrtc=--device-int128
)

add_compile_only_test(test.predefined_macros.cudacc_rtc_int128_short __CUDACC_RTC_INT128__.cu)
target_compile_options(
  test.predefined_macros.cudacc_rtc_int128_short
  PRIVATE -Xnvrtc=-device-int128
)
