//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <span>

// template<size_t Count>
//  constexpr span<element_type, Count> last() const;
//
// constexpr span<element_type, dynamic_extent> last(size_type count) const;
//
//  Requires: Count <= size().

#include <cuda/std/cassert>
#include <cuda/std/span>

#include "test_macros.h"

template <typename Span, size_t Count>
__host__ __device__ constexpr bool testConstexprSpan(Span sp)
{
  assert((noexcept(sp.template last<Count>())));
  assert((noexcept(sp.last(Count))));
  auto s1  = sp.template last<Count>();
  auto s2  = sp.last(Count);
  using S1 = decltype(s1);
  using S2 = decltype(s2);
  ASSERT_SAME_TYPE(typename Span::value_type, typename S1::value_type);
  ASSERT_SAME_TYPE(typename Span::value_type, typename S2::value_type);
  static_assert(S1::extent == Count, "");
  static_assert(S2::extent == cuda::std::dynamic_extent, "");
  return s1.data() == s2.data() && s1.size() == s2.size();
}

template <typename Span, size_t Count>
__host__ __device__ void testRuntimeSpan(Span sp)
{
  assert((noexcept(sp.template last<Count>())));
  assert((noexcept(sp.last(Count))));
  auto s1  = sp.template last<Count>();
  auto s2  = sp.last(Count);
  using S1 = decltype(s1);
  using S2 = decltype(s2);
  ASSERT_SAME_TYPE(typename Span::value_type, typename S1::value_type);
  ASSERT_SAME_TYPE(typename Span::value_type, typename S2::value_type);
  static_assert(S1::extent == Count, "");
  static_assert(S2::extent == cuda::std::dynamic_extent, "");
  assert(s1.data() == s2.data());
  assert(s1.size() == s2.size());
}

STATIC_TEST_GLOBAL_VAR TEST_CONSTEXPR_GLOBAL int carr1[] = {1, 2, 3, 4};
__device__ int arr[]                                     = {5, 6, 7};

int main(int, char**)
{
  {
    using Sp = cuda::std::span<const int>;
    static_assert(testConstexprSpan<Sp, 0>(Sp{}), "");

    static_assert(testConstexprSpan<Sp, 0>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 4>(Sp{carr1}), "");
  }

  {
    using Sp = cuda::std::span<const int, 4>;

    static_assert(testConstexprSpan<Sp, 0>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 1>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 2>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 3>(Sp{carr1}), "");
    static_assert(testConstexprSpan<Sp, 4>(Sp{carr1}), "");
  }

  {
    using Sp = cuda::std::span<int>;
    testRuntimeSpan<Sp, 0>(Sp{});

    testRuntimeSpan<Sp, 0>(Sp{arr});
    testRuntimeSpan<Sp, 1>(Sp{arr});
    testRuntimeSpan<Sp, 2>(Sp{arr});
    testRuntimeSpan<Sp, 3>(Sp{arr});
  }

  {
    using Sp = cuda::std::span<int, 3>;

    testRuntimeSpan<Sp, 0>(Sp{arr});
    testRuntimeSpan<Sp, 1>(Sp{arr});
    testRuntimeSpan<Sp, 2>(Sp{arr});
    testRuntimeSpan<Sp, 3>(Sp{arr});
  }

  return 0;
}
