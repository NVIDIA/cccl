//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__argument_>
#include <cuda/iterator>
#include <cuda/std/array>
#include <cuda/std/complex>
#include <cuda/std/expected>
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

enum class color
{
  red,
  green,
  blue
};

template <class _Tp>
struct element_type_like
{
  using element_type = _Tp;
};

template <class _Tp>
struct range_like
{
  using iterator = _Tp*;
};

template <class _Tp>
struct value_type_like
{
  using value_type = _Tp;
};

struct non_sequence_value
{};

TEST_FUNC void test()
{
  // --- __is_sequence_v / __is_single_value_v ---

  // builtin and class type are not sequences
  static_assert(!cuda::__argument::__is_sequence_v<int>);
  static_assert(!cuda::__argument::__is_sequence_v<color>);
  static_assert(!cuda::__argument::__is_sequence_v<non_sequence_value>);
  static_assert(!cuda::__argument::__is_sequence_v<range_like<int>>);
  static_assert(!cuda::__argument::__is_sequence_v<element_type_like<int>>);
  static_assert(!cuda::__argument::__is_sequence_v<value_type_like<int>>);
  static_assert(!cuda::__argument::__is_sequence_v<cuda::std::complex<float>>);
  static_assert(!cuda::__argument::__is_sequence_v<cuda::std::pair<float, int>>);
  static_assert(!cuda::__argument::__is_sequence_v<cuda::std::tuple<float, int>>);
  static_assert(!cuda::__argument::__is_sequence_v<cuda::std::optional<int>>);
  static_assert(!cuda::__argument::__is_sequence_v<cuda::std::expected<int, int>>);

  // iterators and pointers can be sequences if they are at least random access
  static_assert(cuda::__argument::__is_sequence_v<int*>);
  static_assert(cuda::__argument::__is_sequence_v<const int*>);
  static_assert(cuda::__argument::__is_sequence_v<cuda::counting_iterator<int>>);
  static_assert(!cuda::__argument::__is_sequence_v<bidirectional_iterator<int*>>);

  // ranges and arrays are sequences
  static_assert(cuda::__argument::__is_sequence_v<int[]>);
  static_assert(cuda::__argument::__is_sequence_v<const int[]>);
  static_assert(cuda::__argument::__is_sequence_v<int[42]>);
  static_assert(cuda::__argument::__is_sequence_v<const int[42]>);
  static_assert(cuda::__argument::__is_sequence_v<cuda::std::span<int, 1>>);
  static_assert(cuda::__argument::__is_sequence_v<const cuda::std::span<int, 1>&>);
  static_assert(cuda::__argument::__is_sequence_v<cuda::std::span<int>>);
  static_assert(cuda::__argument::__is_sequence_v<cuda::std::array<int, 3>>);

  // --- __element_type_of_t ---

  static_assert(cuda::std::is_same_v<cuda::__argument::__element_type_of_t<const cuda::std::span<int, 1>&>, int>);
  static_assert(cuda::std::is_same_v<cuda::__argument::__element_type_of_t<int*>, int>);
  static_assert(cuda::std::is_same_v<cuda::__argument::__element_type_of_t<cuda::counting_iterator<int>>, int>);
  static_assert(cuda::std::is_same_v<cuda::__argument::__element_type_of_t<cuda::std::array<int, 3>>, int>);
  static_assert(cuda::std::is_same_v<cuda::__argument::__element_type_of_t<range_like<int>>, int>);
  static_assert(cuda::std::is_same_v<cuda::__argument::__element_type_of_t<element_type_like<int>>, int>);
  static_assert(
    cuda::std::is_same_v<cuda::__argument::__element_type_of_t<cuda::std::mdspan<const int, cuda::std::extents<int, 1>>>,
                         int>);
  static_assert(cuda::std::is_same_v<cuda::__argument::__element_type_of_t<value_type_like<int>>, int>);

  // --- argument_traits: is_deferred ---

  static_assert(!cuda::__argument::__traits<int>::is_deferred);
  static_assert(!cuda::__argument::__traits<cuda::__argument::__immediate<int>>::is_deferred);
  static_assert(!cuda::__argument::__traits<cuda::__argument::__immediate_sequence<cuda::std::span<int>>>::is_deferred);
  static_assert(!cuda::__argument::__traits<cuda::__argument::__constant<42>>::is_deferred);
#if TEST_HAS_CLASS_NTTP
  static_assert(
    !cuda::__argument::__traits<cuda::__argument::__constant_sequence<cuda::std::array<int, 3>{1, 2, 3}>>::is_deferred);
#endif // TEST_HAS_CLASS_NTTP
  static_assert(cuda::__argument::__traits<cuda::__argument::__deferred<cuda::std::span<int, 1>>>::is_deferred);
  static_assert(cuda::__argument::__traits<cuda::__argument::__deferred_sequence<cuda::std::span<int>>>::is_deferred);

  // --- argument_traits: is_single_value ---

  static_assert(cuda::__argument::__traits<int>::is_single_value);
  static_assert(cuda::__argument::__traits<int*>::is_single_value);
  static_assert(cuda::__argument::__traits<cuda::__argument::__immediate<int>>::is_single_value);
  static_assert(cuda::__argument::__traits<cuda::__argument::__immediate<int*>>::is_single_value);
  static_assert(
    cuda::__argument::__traits<cuda::__argument::__immediate<cuda::counting_iterator<int>>>::is_single_value);
  static_assert(
    !cuda::__argument::__traits<cuda::__argument::__immediate_sequence<cuda::std::span<int>>>::is_single_value);
  static_assert(cuda::__argument::__traits<cuda::__argument::__constant<42>>::is_single_value);
#if TEST_HAS_CLASS_NTTP
  static_assert(!cuda::__argument::__traits<
                cuda::__argument::__constant_sequence<cuda::std::array<int, 3>{1, 2, 3}>>::is_single_value);
#endif // TEST_HAS_CLASS_NTTP
  static_assert(cuda::__argument::__traits<cuda::__argument::__deferred<int*>>::is_single_value);
  static_assert(
    !cuda::__argument::__traits<cuda::__argument::__deferred_sequence<cuda::std::span<int>>>::is_single_value);

  // --- argument_traits: value_type ---

  static_assert(cuda::std::is_same_v<cuda::__argument::__traits<int>::value_type, int>);
  static_assert(cuda::std::is_same_v<cuda::__argument::__traits<cuda::__argument::__immediate<int>>::value_type, int>);
  static_assert(cuda::std::is_same_v<
                cuda::__argument::__traits<cuda::__argument::__immediate_sequence<cuda::std::span<int>>>::value_type,
                cuda::std::span<int>>);
  static_assert(cuda::std::is_same_v<cuda::__argument::__traits<cuda::__argument::__constant<42>>::value_type, int>);
#if TEST_HAS_CLASS_NTTP
  static_assert(
    cuda::std::is_same_v<
      cuda::__argument::__traits<cuda::__argument::__constant_sequence<cuda::std::array<int, 3>{1, 2, 3}>>::value_type,
      cuda::std::array<int, 3>>);
#endif // TEST_HAS_CLASS_NTTP

  // --- argument_traits: lowest / highest ---

  static_assert(cuda::__argument::__traits<int>::lowest == cuda::std::numeric_limits<int>::lowest());
  static_assert(cuda::__argument::__traits<int>::highest == (cuda::std::numeric_limits<int>::max)());
  static_assert(cuda::__argument::__traits<const int>::lowest == cuda::std::numeric_limits<int>::lowest());
  static_assert(cuda::__argument::__traits<int&>::highest == (cuda::std::numeric_limits<int>::max)());
  static_assert(cuda::__argument::__traits<float>::lowest == cuda::std::numeric_limits<float>::lowest());
  static_assert(cuda::__argument::__traits<float>::highest == (cuda::std::numeric_limits<float>::max)());
  static_assert(
    cuda::__argument::__traits<const cuda::__argument::__immediate<int, cuda::__argument::__static_bounds<1, 8>>>::lowest
    == 1);
  static_assert(
    cuda::__argument::__traits<cuda::__argument::__immediate<int, cuda::__argument::__static_bounds<1, 8>>&>::highest
    == 8);
  static_assert(
    cuda::__argument::__traits<
      cuda::__argument::__immediate_sequence<cuda::std::span<int>, cuda::__argument::__static_bounds<1, 8>>>::highest
    == 8);
#if TEST_HAS_CLASS_NTTP
  static_assert(
    cuda::__argument::__traits<cuda::__argument::__constant_sequence<cuda::std::array<int, 3>{3, 1, 2}>>::lowest == 1);
  static_assert(
    cuda::__argument::__traits<cuda::__argument::__constant_sequence<cuda::std::array<int, 3>{3, 1, 2}>>::highest == 3);
#endif // TEST_HAS_CLASS_NTTP

  // --- Free function bounds on plain values ---

  static_assert(cuda::__argument::__lowest_(42) == cuda::std::numeric_limits<int>::lowest());
  static_assert(cuda::__argument::__highest_(42) == (cuda::std::numeric_limits<int>::max)());
  static_assert(cuda::__argument::__lowest_(1.0f) == cuda::std::numeric_limits<float>::lowest());
  static_assert(cuda::__argument::__highest_(1.0f) == (cuda::std::numeric_limits<float>::max)());

  // --- Scalar and sequence wrappers expose distinct single-value traits ---

  static_assert(cuda::__argument::__traits<cuda::__argument::__constant<42>>::is_single_value);
  static_assert(cuda::__argument::__traits<cuda::__argument::__immediate<int>>::is_single_value);
  static_assert(
    !cuda::__argument::__traits<cuda::__argument::__immediate_sequence<cuda::std::span<int>>>::is_single_value);
#if TEST_HAS_CLASS_NTTP
  static_assert(!cuda::__argument::__traits<
                cuda::__argument::__constant_sequence<cuda::std::array<int, 3>{1, 2, 3}>>::is_single_value);
#endif // TEST_HAS_CLASS_NTTP
}

int main(int, char**)
{
  test();
  return 0;
}
