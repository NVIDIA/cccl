//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/argument>
#include <cuda/std/array>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct non_sequence_value
{
  int payload;
};

TEST_FUNC void test()
{
  // Basic value
  {
    constexpr auto sa = cuda::args::constant<42>{};
    static_assert(cuda::args::__unwrap(sa) == 42);
    static_assert(cuda::std::is_same_v<decltype(sa)::value_type, int>);
  }

  // Different types
  {
    constexpr auto sa_long = cuda::args::constant<100L>{};
    static_assert(cuda::args::__unwrap(sa_long) == 100L);
    static_assert(cuda::std::is_same_v<decltype(sa_long)::value_type, long>);

    constexpr auto sa_float = cuda::args::constant<10, float>{};
    static_assert(cuda::args::__unwrap(sa_float) == 10.0f);
    static_assert(cuda::std::is_same_v<decltype(sa_float)::value_type, float>);
    static_assert(cuda::std::is_same_v<decltype(cuda::args::__unwrap(sa_float)), float>);
  }

  // Negative value
  {
    constexpr auto sa_neg = cuda::args::constant<-1>{};
    static_assert(cuda::args::__unwrap(sa_neg) == -1);
  }

#if TEST_HAS_CLASS_NTTP
  // Non-sequence values are accepted without scalar-only restrictions
  {
    constexpr auto sa = cuda::args::constant<non_sequence_value{7}>{};
    static_assert(cuda::args::__unwrap(sa).payload == 7);
  }
#endif // TEST_HAS_CLASS_NTTP

#if TEST_HAS_CLASS_NTTP
  // Array sequence
  {
    constexpr auto sa_arr = cuda::args::__constant_sequence<cuda::std::array<int, 3>{128, 256, 512}>{};
    static_assert(cuda::args::__unwrap(sa_arr)[0] == 128);
    static_assert(cuda::args::__unwrap(sa_arr)[1] == 256);
    static_assert(cuda::args::__unwrap(sa_arr)[2] == 512);
    static_assert(cuda::std::is_same_v<decltype(sa_arr)::value_type, cuda::std::array<int, 3>>);
  }
#endif // TEST_HAS_CLASS_NTTP

  // Bounds: scalar
  {
    constexpr auto sa = cuda::args::constant<42>{};
    static_assert(cuda::args::__lowest_(sa) == 42);
    static_assert(cuda::args::__highest_(sa) == 42);
  }

#if TEST_HAS_CLASS_NTTP
  // Bounds: array sequence computes lowest/highest of elements
  {
    constexpr auto sa = cuda::args::__constant_sequence<cuda::std::array<int, 3>{128, 256, 512}>{};
    static_assert(cuda::args::__lowest_(sa) == 128);
    static_assert(cuda::args::__highest_(sa) == 512);
  }
#endif // TEST_HAS_CLASS_NTTP

#if TEST_HAS_CLASS_NTTP
  // Bounds: empty array sequence has unconstrained element bounds
  {
    constexpr auto sa = cuda::args::__constant_sequence<cuda::std::array<int, 0>{}>{};
    static_assert(cuda::args::__lowest_(sa) == cuda::std::numeric_limits<int>::lowest());
    static_assert(cuda::args::__highest_(sa) == (cuda::std::numeric_limits<int>::max)());
  }
#endif // TEST_HAS_CLASS_NTTP

  // Traits
  {
    using traits = cuda::args::__traits<cuda::args::constant<42>>;
    static_assert(!traits::is_deferred);
    static_assert(traits::is_constant);
    static_assert(traits::is_single_value);
    static_assert(cuda::std::is_same_v<traits::value_type, int>);
    static_assert(traits::lowest == 42);
    static_assert(traits::highest == 42);
  }

  // Traits: explicit constant value type
  {
    using traits = cuda::args::__traits<cuda::args::constant<10, float>>;
    static_assert(!traits::is_deferred);
    static_assert(traits::is_constant);
    static_assert(traits::is_single_value);
    static_assert(cuda::std::is_same_v<traits::value_type, float>);
    static_assert(cuda::std::is_same_v<traits::element_type, float>);
    static_assert(traits::lowest == 10.0f);
    static_assert(traits::highest == 10.0f);
  }

#if TEST_HAS_CLASS_NTTP
  // Sequence traits
  {
    using traits = cuda::args::__traits<cuda::args::__constant_sequence<cuda::std::array<int, 3>{1, 2, 3}>>;
    static_assert(traits::is_constant);
    static_assert(!traits::is_deferred);
    static_assert(!traits::is_single_value);
    static_assert(cuda::std::is_same_v<traits::value_type, cuda::std::array<int, 3>>);
    static_assert(cuda::std::is_same_v<traits::element_type, int>);
  }
#endif // TEST_HAS_CLASS_NTTP

  // Single value: scalar is single, sequence is not
  {
    static_assert(!cuda::args::__is_sequence_v<cuda::args::__traits<cuda::args::constant<42>>::value_type>);
#if TEST_HAS_CLASS_NTTP
    static_assert(
      !cuda::args::__traits<cuda::args::__constant_sequence<cuda::std::array<int, 3>{1, 2, 3}>>::is_single_value);
#endif // TEST_HAS_CLASS_NTTP
  }

  // Unwrap: scalar
  {
    constexpr auto sa  = cuda::args::constant<42>{};
    constexpr auto val = cuda::args::__unwrap(sa);
    static_assert(val == 42);
  }

  // Unwrap: scalar with explicit value type
  {
    constexpr auto sa  = cuda::args::constant<10, float>{};
    constexpr auto val = cuda::args::__unwrap(sa);
    static_assert(val == 10.0f);
    static_assert(cuda::std::is_same_v<decltype(val), const float>);
  }

#if TEST_HAS_CLASS_NTTP
  // Unwrap: sequence
  {
    constexpr auto sa  = cuda::args::__constant_sequence<cuda::std::array<int, 3>{10, 20, 30}>{};
    constexpr auto val = cuda::args::__unwrap(sa);
    static_assert(val[0] == 10);
    static_assert(val[2] == 30);
  }
#endif // TEST_HAS_CLASS_NTTP
}

int main(int, char**)
{
  test();
  return 0;
}
