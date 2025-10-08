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
  T arrayA[6] = {};
  T* firstA   = arrayA + 1;
  T* lastA    = arrayA + 5;
  test_in_range(firstA, lastA);

  T arrayB[7] = {};
  T* firstB   = arrayB + 1;
  T* lastB    = arrayB + 7;
  test_in_range(firstB, lastB);
  assert(!cuda::ptr_in_range(firstB, firstA, lastA));
  assert(!cuda::ptr_in_range(lastB, firstA, lastA));
  assert(!cuda::ptr_in_range(firstA, firstB, lastB));
  assert(!cuda::ptr_in_range(lastA, firstB, lastB));

  T* arrayC = new T[6]{};
  T* firstC = arrayC + 1;
  T* lastC  = arrayC + 5;
  test_in_range(firstC, lastC);
  assert(!cuda::ptr_in_range(firstC, firstA, lastA));
  assert(!cuda::ptr_in_range(lastC, firstA, lastA));
  assert(!cuda::ptr_in_range(firstA, firstC, lastC));
  assert(!cuda::ptr_in_range(lastA, firstC, lastC));
  delete[] arrayC;
}

template <typename T>
__host__ __device__ void test_void_variants()
{
  T arrayA[6] = {};
  T* firstA   = arrayA + 1;
  T* lastA    = arrayA + 5;
  test_in_range(static_cast<void*>(firstA), static_cast<void*>(lastA));
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

__host__ __device__ constexpr bool constexpr_test()
{
  constexpr int array[5] = {0, 1, 2, 3, 4};
  assert(cuda::ptr_in_range(array + 1, array, array + 5));
  assert(!cuda::ptr_in_range(array + 5, array, array + 5));
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(constexpr_test());
  return 0;
}
