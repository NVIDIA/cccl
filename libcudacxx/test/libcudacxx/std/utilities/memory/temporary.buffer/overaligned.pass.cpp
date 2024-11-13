//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

// <memory>

// template <class T>
//   pair<T*, ptrdiff_t>
//   get_temporary_buffer(ptrdiff_t n);
//
// template <class T>
//   void
//   return_temporary_buffer(T* p);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/utility>

#include "test_macros.h"

#if defined(TEST_COMPILER_MSVC)
#  pragma warning(disable : 4324) // structure was padded due to alignment specifier
#endif // TEST_COMPILER_MSVC

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
