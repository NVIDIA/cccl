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

// Some code bases define `noinline` as a macro. It must not break the NVCC 12.9 to 13.1 workarounds that use
// _CCCL_NOINLINE.
#define noinline "noinline must not be expanded inside CCCL headers"

#include <cuda/memory>

#ifndef noinline
#  error "<cuda/memory> must restore the macro"
#endif // !noinline

int main(int, char**)
{
  return 0;
}
