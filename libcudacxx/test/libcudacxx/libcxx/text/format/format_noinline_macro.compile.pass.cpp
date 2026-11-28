//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// NVRTC does not support #pragma push_macro and #pragma pop_macro.
// UNSUPPORTED: nvrtc

// Some code bases define `noinline` as a macro. It must not break the formatting functions that use
// _CCCL_NOINLINE_DEVICE.
// libstdc++ spells `__attribute__((__noinline__))`, which the `__noinline__` macro of the CUDA Toolkit turns into
// `__attribute__((noinline))`. So include the host headers that do so before defining the macro.
#if __has_include(<memory>)
#  include <memory>
#  include <string>
#endif // __has_include(<memory>)
#if __has_include(<version>)
#  include <version>
#endif // __has_include(<version>)
#if __cpp_lib_format >= 201907L
#  include <format>
#endif // __cpp_lib_format >= 201907L

#define noinline "noinline must not be expanded inside CCCL headers"

#include <cuda/std/__format_>

#ifndef noinline
#  error "<cuda/std/__format_> must restore the macro"
#endif // !noinline

int main(int, char**)
{
  return 0;
}
