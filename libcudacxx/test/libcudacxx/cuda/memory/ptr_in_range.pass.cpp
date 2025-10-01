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
#include <cuda/std/type_traits>

template <typename Pointer>
__host__ __device__ void test_in_range([[maybe_unused]] Pointer first, [[maybe_unused]] Pointer last)
{
  assert(cuda::ptr_in_range(first, first, last));
  assert(cuda::ptr_in_range(first + 1, first, last));
  assert(cuda::ptr_in_range(last - 1, first, last));
  assert(!cuda::ptr_in_range(last, first, last));
}

template <typename T>
__host__ __device__ void test_variants()
{
  T values[6] = {};
  T* first    = values + 1;
  T* last     = values + 5;
  test_in_range(first, last);
}

__host__ __device__ bool test()
{
  constexpr auto nullptr_int = static_cast<int*>(nullptr);
  static_assert(noexcept(cuda::ptr_in_range(nullptr_int, nullptr_int, nullptr_int)));
  using ret_type = decltype(cuda::ptr_in_range(nullptr_int, nullptr_int, nullptr_int));
  static_assert(::cuda::std::is_same_v<bool, ret_type>);

  test_variants<int>();
  test_variants<const int>();
  return true;
}

int main(int, char**)
{
  assert(test());
  return 0;
}
