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
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ constexpr void test()
{
  using inplace_vector = cuda::std::inplace_vector<T, 42>;
  inplace_vector range{T(1), T(1337), T(42), T(12), T(0), T(-1)};
  const inplace_vector const_range{T(0), T(42), T(1337), T(42), T(5), T(-42)};

  const auto begin = range.begin();
  static_assert(cuda::std::is_same<decltype(begin), const typename inplace_vector::iterator>::value, "");
  assert(*begin == T(1));

  const auto cbegin = range.cbegin();
  static_assert(cuda::std::is_same<decltype(cbegin), const typename inplace_vector::const_iterator>::value, "");
  assert(*cbegin == T(1));

  const auto const_begin = const_range.begin();
  static_assert(cuda::std::is_same<decltype(const_begin), const typename inplace_vector::const_iterator>::value, "");
  assert(*const_begin == T(0));

  const auto end = range.end();
  static_assert(cuda::std::is_same<decltype(end), const typename inplace_vector::iterator>::value, "");
  assert(*cuda::std::prev(end) == T(-1));

  const auto cend = range.cend();
  static_assert(cuda::std::is_same<decltype(cend), const typename inplace_vector::const_iterator>::value, "");
  assert(*cuda::std::prev(cend) == T(-1));

  const auto const_end = const_range.end();
  static_assert(cuda::std::is_same<decltype(const_end), const typename inplace_vector::const_iterator>::value, "");
  assert(*cuda::std::prev(const_end) == T(-42));

  const auto rbegin = range.rbegin();
  static_assert(cuda::std::is_same<decltype(rbegin), const typename inplace_vector::reverse_iterator>::value, "");
  assert(*rbegin == T(-1));

  const auto crbegin = range.crbegin();
  static_assert(cuda::std::is_same<decltype(crbegin), const typename inplace_vector::const_reverse_iterator>::value,
                "");
  assert(*crbegin == T(-1));

  const auto const_rbegin = const_range.rbegin();
  static_assert(
    cuda::std::is_same<decltype(const_rbegin), const typename inplace_vector::const_reverse_iterator>::value, "");
  assert(*const_rbegin == T(-42));

  const auto rend = range.rend();
  static_assert(cuda::std::is_same<decltype(rend), const typename inplace_vector::reverse_iterator>::value, "");
  assert(*cuda::std::prev(rend) == T(1));

  const auto crend = range.crend();
  static_assert(cuda::std::is_same<decltype(crend), const typename inplace_vector::const_reverse_iterator>::value, "");
  assert(*cuda::std::prev(crend) == T(1));

  const auto const_rend = const_range.rend();
  static_assert(cuda::std::is_same<decltype(const_rend), const typename inplace_vector::const_reverse_iterator>::value,
                "");
  assert(*cuda::std::prev(const_rend) == T(0));
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
