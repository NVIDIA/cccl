//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// Test iterator category and iterator concepts.

// using index-type = conditional_t<same_as<Bound, unreachable_sentinel_t>, ptrdiff_t, Bound>;
// using iterator_concept = random_access_iterator_tag;
// using iterator_category = random_access_iterator_tag;
// using value_type = T;
// using difference_type = see below:
// If is-signed-integer-like<index-type> is true, the member typedef-name difference_type denotes
// index-type. Otherwise, it denotes IOTA-DIFF-T(index-type).

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/cstdint>
#include <cuda/std/ranges>
#include <cuda/std/type_traits>

__host__ __device__ constexpr bool test()
{
  // unbound
  {
    using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::repeat_view<int>>;
    static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<Iter::value_type, int>);
    static_assert(cuda::std::same_as<Iter::difference_type, ptrdiff_t>);
    static_assert(cuda::std::is_signed_v<Iter::difference_type>);
  }

  // bound
  {
    {
      using Iter = cuda::std::ranges::iterator_t<const cuda::std::ranges::repeat_view<int, cuda::std::int8_t>>;
      static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::value_type, int>);
      static_assert(cuda::std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) == sizeof(cuda::std::int8_t));
    }

    {
      using Iter = cuda::std::ranges::iterator_t<const cuda::std::ranges::repeat_view<int, cuda::std::uint8_t>>;
      static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::value_type, int>);
      static_assert(cuda::std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) > sizeof(cuda::std::uint8_t));
    }

    {
      using Iter = cuda::std::ranges::iterator_t<const cuda::std::ranges::repeat_view<int, cuda::std::int16_t>>;
      static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::value_type, int>);
      static_assert(cuda::std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) == sizeof(cuda::std::int16_t));
    }

    {
      using Iter = cuda::std::ranges::iterator_t<const cuda::std::ranges::repeat_view<int, cuda::std::uint16_t>>;
      static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::value_type, int>);
      static_assert(cuda::std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) > sizeof(cuda::std::uint16_t));
    }

    {
      using Iter = cuda::std::ranges::iterator_t<const cuda::std::ranges::repeat_view<int, cuda::std::int32_t>>;
      static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::value_type, int>);
      static_assert(cuda::std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) == sizeof(cuda::std::int32_t));
    }

    {
      using Iter = cuda::std::ranges::iterator_t<const cuda::std::ranges::repeat_view<int, cuda::std::uint32_t>>;
      static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::value_type, int>);
      static_assert(cuda::std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) > sizeof(cuda::std::uint32_t));
    }

    {
      using Iter = cuda::std::ranges::iterator_t<const cuda::std::ranges::repeat_view<int, cuda::std::int64_t>>;
      static_assert(cuda::std::same_as<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::iterator_category, cuda::std::random_access_iterator_tag>);
      static_assert(cuda::std::same_as<Iter::value_type, int>);
      static_assert(cuda::std::is_signed_v<Iter::difference_type>);
      static_assert(sizeof(Iter::difference_type) == sizeof(cuda::std::int64_t));
    }
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
