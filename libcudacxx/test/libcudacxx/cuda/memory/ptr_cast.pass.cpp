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
  uintptr_t ptr_int = 12;
  auto ptr          = reinterpret_cast<char*>(ptr_int);
  assert(cuda::ptr_cast<uint16_t>(ptr) == ptr);
  assert(cuda::ptr_cast<int>(ptr) == ptr);
  assert(cuda::ptr_cast<uint64_t>(ptr) == ptr);
  static_cast(cuda::std::is_same_v<int*, decltype(cuda::ptr_cast<int>(ptr))>);
  static_cast(cuda::std::is_same_v<const int*, decltype(cuda::ptr_cast<int>((const int*) ptr))>);
}

int main(int, char**)
{
  assert(test());
  return 0;
}
