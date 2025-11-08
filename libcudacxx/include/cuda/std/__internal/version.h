//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

#ifndef _CUDA_STD___INTERNAL_VERSION_H
#define _CUDA_STD___INTERNAL_VERSION_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/version.h> // IWYU pragma: export

#define _LIBCUDACXX_CUDA_API_VERSION       CCCL_VERSION
#define _LIBCUDACXX_CUDA_API_VERSION_MAJOR CCCL_MAJOR_VERSION
#define _LIBCUDACXX_CUDA_API_VERSION_MINOR CCCL_MINOR_VERSION
#define _LIBCUDACXX_CUDA_API_VERSION_PATCH CCCL_PATCH_VERSION

#ifndef _LIBCUDACXX_CUDA_ABI_VERSION_LATEST
#  define _LIBCUDACXX_CUDA_ABI_VERSION_LATEST 4
#endif

#ifdef _LIBCUDACXX_CUDA_ABI_VERSION
#  if _LIBCUDACXX_CUDA_ABI_VERSION != 4
#    error Unsupported libcu++ ABI version requested. Only version 4 is allowed.
#  endif
#else
#  define _LIBCUDACXX_CUDA_ABI_VERSION _LIBCUDACXX_CUDA_ABI_VERSION_LATEST
#endif

#if (_LIBCUDACXX_CUDA_ABI_VERSION < 4) && !defined(LIBCUDACXX_IGNORE_DEPRECATED_ABI)
#  error "libcu++ ABIs older than version 4 are deprecated, define LIBCUDACXX_IGNORE_DEPRECATED_ABI to ignore"
#endif

#ifdef _LIBCUDACXX_PIPELINE_ASSUMED_ABI_VERSION
#  if _LIBCUDACXX_PIPELINE_ASSUMED_ABI_VERSION != _LIBCUDACXX_CUDA_ABI_VERSION
#    error cuda_pipeline.h has assumed a different libcu++ ABI version than provided by this library. To fix this, please include a libcu++ header before including cuda_pipeline.h, or upgrade to a version of the toolkit this version of libcu++ shipped in.
#  endif
#endif

#endif // _CUDA_STD___INTERNAL_VERSION_H
