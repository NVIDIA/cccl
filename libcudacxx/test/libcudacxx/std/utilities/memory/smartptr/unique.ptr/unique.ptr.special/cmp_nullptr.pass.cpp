//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// <memory>

// unique_ptr

// template <class T, class D>
//     constexpr bool operator==(const unique_ptr<T, D>& x, nullptr_t) noexcept; // constexpr since C++23
// template <class T, class D>
//     bool operator==(nullptr_t, const unique_ptr<T, D>& y) noexcept;           // removed in C++20
// template <class T, class D>
//     bool operator!=(const unique_ptr<T, D>& x, nullptr_t) noexcept;           // removed in C++20
// template <class T, class D>
//     bool operator!=(nullptr_t, const unique_ptr<T, D>& y) noexcept;           // removed in C++20
// template <class T, class D>
//     constexpr bool operator<(const unique_ptr<T, D>& x, nullptr_t);           // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator<(nullptr_t, const unique_ptr<T, D>& y);           // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator<=(const unique_ptr<T, D>& x, nullptr_t);          // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator<=(nullptr_t, const unique_ptr<T, D>& y);          // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator>(const unique_ptr<T, D>& x, nullptr_t);           // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator>(nullptr_t, const unique_ptr<T, D>& y);           // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator>=(const unique_ptr<T, D>& x, nullptr_t);          // constexpr since C++23
// template <class T, class D>
//     constexpr bool operator>=(nullptr_t, const unique_ptr<T, D>& y);          // constexpr since C++23
// template<class T, class D>
//   requires three_way_comparable<typename unique_ptr<T, D>::pointer>
//   constexpr compare_three_way_result_t<typename unique_ptr<T, D>::pointer>
//     operator<=>(const unique_ptr<T, D>& x, nullptr_t);                        // C++20

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

#if defined(TEST_COMPILER_NVCC) || defined(TEST_COMPILER_NVRTC)
TEST_NV_DIAG_SUPPRESS(3060) // call to __builtin_is_constant_evaluated appearing in a non-constexpr function
#endif // TEST_COMPILER_NVCC || TEST_COMPILER_NVRTC
#if defined(TEST_COMPILER_GCC)
#  pragma GCC diagnostic ignored "-Wtautological-compare"
#elif defined(TEST_COMPILER_CLANG)
#  pragma clang diagnostic ignored "-Wtautological-compare"
#endif

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    AssertEqualityAreNoexcept<cuda::std::unique_ptr<int>, cuda::std::nullptr_t>();
    AssertEqualityAreNoexcept<cuda::std::nullptr_t, cuda::std::unique_ptr<int>>();
    AssertComparisonsReturnBool<cuda::std::unique_ptr<int>, cuda::std::nullptr_t>();
    AssertComparisonsReturnBool<cuda::std::nullptr_t, cuda::std::unique_ptr<int>>();
#if TEST_STD_VER >= 2020 && !defined(TEST_HAS_NO_SPACESHIP_OPERATOR)
    AssertOrderReturn<cuda::std::strong_ordering, cuda::std::unique_ptr<int>, cuda::std::nullptr_t>();
    AssertOrderReturn<cuda::std::strong_ordering, cuda::std::nullptr_t, cuda::std::unique_ptr<int>>();
#endif // TEST_STD_VER >= 2020 && !defined(TEST_HAS_NO_SPACESHIP_OPERATOR)
  }

  const cuda::std::unique_ptr<int> p1(new int(1));
  assert(!(p1 == nullptr));
  assert(!(nullptr == p1));
  // A pointer to allocated storage and a nullptr can't be compared at compile-time
  if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
  {
    assert(!(p1 < nullptr));
    assert((nullptr < p1));
    assert(!(p1 <= nullptr));
    assert((nullptr <= p1));
    assert((p1 > nullptr));
    assert(!(nullptr > p1));
    assert((p1 >= nullptr));
    assert(!(nullptr >= p1));
#if TEST_STD_VER >= 2020 && !defined(TEST_HAS_NO_SPACESHIP_OPERATOR)
    assert((nullptr <=> p1) == cuda::std::strong_ordering::less);
    assert((p1 <=> nullptr) == cuda::std::strong_ordering::greater);
#endif // TEST_STD_VER >= 2020 && !defined(TEST_HAS_NO_SPACESHIP_OPERATOR)
  }

  const cuda::std::unique_ptr<int> p2;
  assert((p2 == nullptr));
  assert((nullptr == p2));
  assert(!(p2 < nullptr));
  assert(!(nullptr < p2));
  assert((p2 <= nullptr));
  assert((nullptr <= p2));
  assert(!(p2 > nullptr));
  assert(!(nullptr > p2));
  assert((p2 >= nullptr));
  assert((nullptr >= p2));
#if TEST_STD_VER >= 2020 && !defined(TEST_HAS_NO_SPACESHIP_OPERATOR)
  assert((nullptr <=> p2) == cuda::std::strong_ordering::equivalent);
#endif // TEST_STD_VER >= 2020 && !defined(TEST_HAS_NO_SPACESHIP_OPERATOR)

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}
