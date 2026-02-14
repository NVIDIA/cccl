//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_IGNORE_DEPRECATED_API

// <memory>

// template <class T>
//   pair<T*, ptrdiff_t>
//   get_temporary_buffer(ptrdiff_t n);
//
// template <class T>
//   void
//   return_temporary_buffer(T* p);

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/memory>
#include <cuda/std/utility>

#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4324) // structure was padded due to alignment specifier

struct alignas(32) A
{
  int field;
};

int main(int, char**)
{
  cuda::std::pair<A*, cuda::std::ptrdiff_t> ip = cuda::std::get_temporary_buffer<A>(5);
  assert(!(ip.first == nullptr) ^ (ip.second == 0));
  assert(reinterpret_cast<cuda::std::uintptr_t>(ip.first) % alignof(A) == 0);
  cuda::std::return_temporary_buffer(ip.first);

  return 0;
}
