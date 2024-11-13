//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11

#include <cuda/std/__algorithm_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/inplace_vector>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ constexpr void test()
{
  constexpr size_t max_capacity = 42ull;
  using inplace_vector          = cuda::std::inplace_vector<T, max_capacity>;
  inplace_vector range{T(1), T(1337), T(42), T(12), T(0), T(-1)};
  const inplace_vector const_range{T(0), T(42), T(1337), T(42), T(5), T(-42)};

  const auto empty = range.empty();
  static_assert(cuda::std::is_same<decltype(empty), const bool>::value, "");
  assert(!empty);

  const auto const_empty = const_range.empty();
  static_assert(cuda::std::is_same<decltype(const_empty), const bool>::value, "");
  assert(!const_empty);

  const auto size = range.size();
  static_assert(cuda::std::is_same<decltype(size), const typename inplace_vector::size_type>::value, "");
  assert(size == 6);

  const auto const_size = const_range.size();
  static_assert(cuda::std::is_same<decltype(const_size), const typename inplace_vector::size_type>::value, "");
  assert(const_size == 6);

  const auto max_size = range.max_size();
  static_assert(cuda::std::is_same<decltype(max_size), const typename inplace_vector::size_type>::value, "");
  assert(max_size == max_capacity);

  const auto const_max_size = const_range.max_size();
  static_assert(cuda::std::is_same<decltype(const_max_size), const typename inplace_vector::size_type>::value, "");
  assert(const_max_size == max_capacity);

  const auto capacity = range.capacity();
  static_assert(cuda::std::is_same<decltype(capacity), const typename inplace_vector::size_type>::value, "");
  assert(capacity == max_capacity);

  const auto const_capacity = const_range.capacity();
  static_assert(cuda::std::is_same<decltype(const_capacity), const typename inplace_vector::size_type>::value, "");
  assert(const_capacity == max_capacity);
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<Trivial>();

  if (!cuda::std::__libcpp_is_constant_evaluated())
  {
    test<NonTrivial>();
    test<NonTrivialDestructor>();
    test<ThrowingDefaultConstruct>();
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

  return 0;
}
