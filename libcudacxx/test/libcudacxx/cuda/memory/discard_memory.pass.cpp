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
#include <cuda/std/cstddef>
#include <cuda/std/cstdlib>

template <typename T>
__device__ __host__ void test_discard_memory()
{
  constexpr cuda::std::size_t n      = 128;
  constexpr cuda::std::size_t nbytes = n * sizeof(T);

  auto ptr = static_cast<volatile T*>(cuda::std::malloc(nbytes));
  assert(ptr != nullptr);

  for (cuda::std::size_t i = 0; i < n; ++i)
  {
    ptr[i] = static_cast<T>(i);
  }

  cuda::discard_memory(ptr, nbytes);

  cuda::std::free(const_cast<T*>(ptr));
}

__device__ __host__ void test()
{
  test_discard_memory<int>();
}

int main(int, char**)
{
  test();
  return 0;
}
