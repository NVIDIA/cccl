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
  { // inplace_vector<T, 0> can be swapped
    using inplace_vector = cuda::std::inplace_vector<T, 0>;
    inplace_vector empty{};
    empty.swap(empty);
    static_assert(noexcept(empty.swap(empty)), "");
    swap(empty, empty);
    static_assert(noexcept(swap(empty, empty)), "");
  }

  { // inplace_vector<T, N> can be swapped
    using inplace_vector = cuda::std::inplace_vector<T, 42>;
    cuda::std::array<T, 6> expected_left{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    cuda::std::array<T, 5> expected_right{T(0), T(42), T(1337), T(42), T(5)};

    inplace_vector left{T(1), T(1337), T(42), T(12), T(0), T(-1)};
    inplace_vector right{T(0), T(42), T(1337), T(42), T(5)};

    left.swap(right);
    constexpr bool nothrow_swap =
      cuda::std::is_nothrow_swappable<T>::value && cuda::std::is_nothrow_move_constructible<T>::value;
    static_assert(noexcept(left.swap(right)) == nothrow_swap, "");
    assert(equal_range(left, expected_right));
    assert(equal_range(right, expected_left));

    swap(left, right);
#if !defined(TEST_COMPILER_MSVC_2017)
    static_assert(noexcept(swap(left, right)) == nothrow_swap, "");
#endif // !TEST_COMPILER_MSVC_2017
    assert(equal_range(left, expected_left));
    assert(equal_range(right, expected_right));
  }
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
    test<ThrowingMoveConstructor>();
    test<ThrowingSwap>();
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
