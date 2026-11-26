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

// Some code bases define `empty_bases` as a macro. It must not break _CCCL_DECLSPEC_EMPTY_BASES, which spells
// __declspec(empty_bases) with MSVC. These headers include every libcu++ header that uses it.
#define empty_bases "empty_bases must not be expanded inside CCCL headers"

#include <cuda/execution>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/mdspan>
#include <cuda/memory_resource>
#include <cuda/std/execution>
#include <cuda/std/mdspan>
#include <cuda/std/ranges>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/utility>

#ifndef empty_bases
#  error "libcu++ headers must restore the macro"
#endif // !empty_bases

int main(int, char**)
{
  return 0;
}
