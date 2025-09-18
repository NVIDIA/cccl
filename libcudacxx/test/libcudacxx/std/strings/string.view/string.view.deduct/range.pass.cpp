//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

//  template<class Range>
//    basic_string_view(Range&&) -> basic_string_view<ranges::range_value_t<Range>>; // C++23

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"
#include "test_iterators.h"

template <class CharT>
__host__ __device__ constexpr void test_range_deduct()
{
  // 1. Test construction of a string_view from an cuda::std::array
  {
    constexpr auto str = TEST_STRLIT(CharT, "test");
    cuda::std::array<CharT, 4> val{str[0], str[1], str[2], str[3]};
    auto sv = cuda::std::basic_string_view(val);
    static_assert(cuda::std::is_same_v<decltype(sv), cuda::std::basic_string_view<CharT>>);
    assert(sv.size() == val.size());
    assert(sv.data() == val.data());
  }

  // 2. Test construction of a string_view from a custom type
  {
    struct Widget
    {
      __host__ __device__ constexpr const CharT* data() const
      {
        return data_;
      }
      __host__ __device__ constexpr contiguous_iterator<const CharT*> begin() const
      {
        return contiguous_iterator<const CharT*>(data());
      }
      __host__ __device__ constexpr contiguous_iterator<const CharT*> end() const
      {
        return contiguous_iterator<const CharT*>(data() + 3);
      }

      const CharT* data_;
    };
    const auto widget_data           = TEST_STRLIT(CharT, "foo");
    cuda::std::basic_string_view bsv = cuda::std::basic_string_view(Widget{widget_data});
    static_assert(cuda::std::is_same_v<decltype(bsv), cuda::std::basic_string_view<CharT>>);
    assert(bsv.size() == 3);
    assert(bsv.data() == widget_data);
  }
}

__host__ __device__ constexpr bool test()
{
  test_range_deduct<char>();
#if _CCCL_HAS_CHAR8_T()
  test_range_deduct<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_range_deduct<char16_t>();
  test_range_deduct<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test_range_deduct<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
