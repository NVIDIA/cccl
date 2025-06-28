//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

__host__ __device__ bool test()
{
  uintptr_t ptr_int = 16;
  auto ptr          = reinterpret_cast<char*>(ptr_int);
  assert(cuda::ptr_cast<uint16_t>(ptr) == (uint16_t*) ptr);
  assert(cuda::ptr_cast<int>(ptr) == (int*) ptr);
  assert(cuda::ptr_cast<uint64_t>(ptr) == (uint64_t*) ptr);
  static_assert(cuda::std::is_same_v<int*, decltype(cuda::ptr_cast<int>(ptr))>);

  auto const_ptr = reinterpret_cast<const char*>(ptr_int);
  assert(cuda::ptr_cast<uint16_t>(const_ptr) == (const uint16_t*) ptr);
  static_assert(cuda::std::is_same_v<const int*, decltype(cuda::ptr_cast<int>(const_ptr))>);
  return true;
}

int main(int, char**)
{
  assert(test());
  return 0;
}
