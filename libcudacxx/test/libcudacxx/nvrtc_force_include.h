//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_NVRTC_FORCE_INCLUDE_H
#define _LIBCUDACXX_NVRTC_FORCE_INCLUDE_H

#if defined(__CUDACC_RTC__)
#  define TEST_NVRTC
#  define TEST_NVRTC_VER_MAJOR __CUDACC_VER_MAJOR__
#  define TEST_NVRTC_VER_MINOR __CUDACC_VER_MINOR__
#endif

// Disable once this bug is fixed
#if defined(TEST_NVRTC) && (TEST_NVRTC_VER_MAJOR >= 13)
#  define TEST_NVRTC_DISABLE_VIRTUAL_DEFAULT_DTOR_STRUCT
#  define TEST_NVRTC_VIRTUAL_DEFAULT_DTOR_ANNOTATION __host__ __device__
#else
#  define TEST_NVRTC_VIRTUAL_DEFAULT_DTOR_ANNOTATION
#endif

#endif // _LIBCUDACXX_NVRTC_FORCE_INCLUDE_H
