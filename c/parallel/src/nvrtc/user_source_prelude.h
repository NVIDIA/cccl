//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>
#include <string_view>

// Source placed in front of the C++ sources of user operators and iterators before NVRTC compiles them into a
// translation unit of their own (see with_user_source_prelude below). The kernel translation unit does not get it.
//
// The CUDA 12.4 Update 1 headers (cuda_bf16.h 12.4.127) moved the definition of __half::__half(__nv_bfloat16) from
// cuda_bf16.hpp, which NVRTC 12.3+ never parses, into cuda_bf16.h. It is spelled with __CUDA_BF16_FORCEINLINE__,
// which those NVRTC versions expand to nothing, so every translation unit that includes cuda_bf16.h defines the
// constructor with external linkage, and linking the kernel with any other such unit fails with
// "'_ZN6__halfC1E13__nv_bfloat16': symbol multiply defined!" (NVIDIA/cccl#11885). CUDA 12.5 added an explicit
// `inline`.
//
// Including cuda_bf16.h with __CUDA_NO_HALF_CONVERSIONS__ defined skips that one definition; it is the only thing in
// cuda_bf16.h guarded by the macro. cuda_fp16.h is included first, while the macro is still undefined, so __half keeps
// all of its conversions, including the declaration of this constructor, and the CCCL headers, which also read the
// macro, are never parsed while it is defined. A call to the constructor from such a unit then resolves at link time
// to the one definition left in the link: the kernel's, which includes the header normally. That is only safe on CUDA
// 12.4, where the kernel's definition is either external (Update 1) or comes from the NVRTC built-ins library (GA);
// from 12.5 on the kernel's definition is inline and, being unused there, need not be emitted, so the prelude is
// limited to NVRTC 12.4 and every other version is left untouched. A unit that defines the macro itself keeps that
// behavior. `#line 1` keeps the line numbers in NVRTC diagnostics those of the user's source.
inline constexpr std::string_view nvrtc_user_source_prelude = R"XXX(
#if defined(__CUDACC_RTC__) && (__CUDACC_VER_MAJOR__ == 12) && (__CUDACC_VER_MINOR__ == 4) \
  && !defined(__CUDA_NO_HALF_CONVERSIONS__) && __has_include(<cuda_fp16.h>) && __has_include(<cuda_bf16.h>)
#  include <cuda_fp16.h>
#  define __CUDA_NO_HALF_CONVERSIONS__
#  include <cuda_bf16.h>
#  undef __CUDA_NO_HALF_CONVERSIONS__
#endif
#line 1
)XXX";

// Returns `source` with nvrtc_user_source_prelude in front of it. Every C++ source compiled into a translation unit
// other than the kernel's must go through this.
inline std::string with_user_source_prelude(std::string_view source)
{
  std::string result;
  result.reserve(nvrtc_user_source_prelude.size() + source.size());
  result.append(nvrtc_user_source_prelude).append(source);
  return result;
}
