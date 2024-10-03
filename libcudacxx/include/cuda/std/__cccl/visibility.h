//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_VISIBILITY_H
#define __CCCL_VISIBILITY_H

#ifndef _CUDA__CCCL_CONFIG
#  error "<__cccl/visibility.h> should only be included in from <cuda/__cccl_config>"
#endif // _CUDA__CCCL_CONFIG

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

// We want to ensure that all warning emmiting from this header are supressed
#if defined(_CCCL_FORCE_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_FORCE_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_FORCE_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/attributes.h>
#include <cuda/std/__cccl/execution_space.h>

// For unknown reasons, nvc++ need to selectively disable this warning
// We do not want to use our usual macro because that would have push / pop semantics
#if defined(_CCCL_COMPILER_NVHPC)
#  pragma nv_diag_suppress 1407
#endif // _CCCL_COMPILER_NVHPC

// Enable us to hide kernels
#if defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_VISIBILITY_HIDDEN
#else // ^^^ _CCCL_COMPILER_NVRTC ^^^ / vvv _CCCL_COMPILER_NVRTC vvv
#  define _CCCL_VISIBILITY_HIDDEN __attribute__((__visibility__("hidden")))
#endif // !_CCCL_COMPILER_NVRTC

#if defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_VISIBILITY_DEFAULT __declspec(dllimport)
#elif defined(_CCCL_COMPILER_NVRTC) // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv _CCCL_COMPILER_NVRTC vvv
#  define _CCCL_VISIBILITY_DEFAULT
#else // ^^^ _CCCL_COMPILER_NVRTC ^^^ / vvv !_CCCL_COMPILER_NVRTC vvv
#  define _CCCL_VISIBILITY_DEFAULT __attribute__((__visibility__("default")))
#endif // !_CCCL_COMPILER_NVRTC

#if defined(_CCCL_COMPILER_MSVC) || defined(_CCCL_COMPILER_NVRTC)
#  define _CCCL_TYPE_VISIBILITY_DEFAULT
#elif _CCCL_HAS_ATTRIBUTE(__type_visibility__)
#  define _CCCL_TYPE_VISIBILITY_DEFAULT __attribute__((__type_visibility__("default")))
#else // ^^^ _CCCL_HAS_ATTRIBUTE(__type_visibility__) ^^^ / vvv !_CCCL_HAS_ATTRIBUTE(__type_visibility__) vvv
#  define _CCCL_TYPE_VISIBILITY_DEFAULT _CCCL_VISIBILITY_DEFAULT
#endif // !_CCCL_COMPILER_NVRTC

#if defined(_CCCL_COMPILER_MSVC)
#  define _CCCL_FORCEINLINE __forceinline
#else // ^^^ _CCCL_COMPILER_MSVC ^^^ / vvv _CCCL_COMPILER_MSVC vvv
#  define _CCCL_FORCEINLINE __inline__ __attribute__((__always_inline__))
#endif // !_CCCL_COMPILER_MSVC

#if _CCCL_HAS_ATTRIBUTE(exclude_from_explicit_instantiation)
#  define _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION __attribute__((exclude_from_explicit_instantiation))
#else // ^^^ exclude_from_explicit_instantiation ^^^ / vvv !exclude_from_explicit_instantiation vvv
// NVCC complains mightily about being unable to inline functions if we use _CCCL_FORCEINLINE here
#  define _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#endif // !exclude_from_explicit_instantiation

#if defined(_CCCL_COMPILER_ICC) // ICC has issues with visibility attributes on symbols with internal linkage
#  define _CCCL_HIDE_FROM_ABI inline
#else // ^^^ _CCCL_COMPILER_ICC ^^^ / vvv !_CCCL_COMPILER_ICC vvv
#  define _CCCL_HIDE_FROM_ABI _CCCL_VISIBILITY_HIDDEN _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION inline
#endif // !_CCCL_COMPILER_ICC

//! Defined here to supress any warnings from the definition
#define _LIBCUDACXX_HIDE_FROM_ABI _CCCL_HIDE_FROM_ABI _CCCL_HOST_DEVICE

#if !defined(CCCL_DETAIL_KERNEL_ATTRIBUTES)
#  define CCCL_DETAIL_KERNEL_ATTRIBUTES __global__ _CCCL_VISIBILITY_HIDDEN
#endif // !CCCL_DETAIL_KERNEL_ATTRIBUTES

#endif // __CCCL_VISIBILITY_H
