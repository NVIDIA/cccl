//===----------------------------------------------------------------------===//
//
// Part of nvrtcc in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#if !defined(__NVRTCC__)
#  error "missing __NVRTCC__ definition"
#endif // !__NVRTCC__

#if defined(__NVCC__) != !defined(__CUDACC_RTC__)
#  error "__NVCC__ and __CUDACC_RTC__ should never be defined at the same time"
#endif // __NVCC__ != !__CUDACC_RTC__

#if !defined(__CUDACC__)
#  error "missing __CUDACC__ definition"
#endif // !__CUDACC__

#if !defined(__CUDA_ARCH_LIST__)
#  error "missing __CUDA_ARCH_LIST__ definition."
#endif // !__CUDA_ARCH_LIST__

#if !defined(__CUDACC_VER_MAJOR__)
#  error "missing __CUDACC_VER_MAJOR__ definition"
#endif // !__CUDACC_VER_MAJOR__

#if !defined(__CUDACC_VER_MINOR__)
#  error "missing __CUDACC_VER_MINOR__ definition"
#endif // !__CUDACC_VER_MINOR__

#if !defined(__CUDACC_VER_BUILD__)
#  error "missing __CUDACC_VER_BUILD__ definition"
#endif // !__CUDACC_VER_BUILD__

#if !defined(__NVCC_DIAG_PRAGMA_SUPPORT__)
#  error "missing __NVCC_DIAG_PRAGMA_SUPPORT__ definition"
#endif // !__NVCC_DIAG_PRAGMA_SUPPORT__

#if defined(__CUDACC_DEBUG__) != defined(EXPECT_CUDACC_DEBUG)
#  error "__CUDACC_DEBUG__ must match EXPECT_CUDACC_DEBUG definition."
#endif // __CUDACC_DEBUG__ != EXPECT_CUDACC_DEBUG

#if defined(__CUDACC_EWP__) != defined(EXPECT_CUDACC_EWP)
#  error "__CUDACC_EWP__ must match EXPECT_CUDACC_EWP definition."
#endif // __CUDACC_EWP__ != EXPECT_CUDACC_EWP

#if defined(__CUDACC_RDC__) != defined(EXPECT_CUDACC_RDC)
#  error "__CUDACC_RDC__ must match EXPECT_CUDACC_RDC definition."
#endif // __CUDACC_RDC__ != EXPECT_CUDACC_RDC

#if defined(__CUDACC_RTC_INT128__) != defined(EXPECT_CUDACC_RTC_INT128)
#  error "__CUDACC_RTC_INT128__ must match EXPECT_CUDACC_RTC_INT128 definition."
#endif // __CUDACC_RTC_INT128__ != EXPECT_CUDACC_RTC_INT128

#if defined(__CUDACC_RTC_FLOAT128__) != defined(EXPECT_CUDACC_RTC_FLOAT128)
#  error "__CUDACC_RTC_FLOAT128__ must match EXPECT_CUDACC_RTC_FLOAT128 definition."
#endif // __CUDACC_RTC_FLOAT128__ != EXPECT_CUDACC_RTC_FLOAT128
