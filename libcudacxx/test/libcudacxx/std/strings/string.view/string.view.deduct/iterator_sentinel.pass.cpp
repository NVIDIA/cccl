//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

//  template <contiguous_iterator _It, sized_sentinel_for<_It> _End>
//    basic_string_view(_It, _End) -> basic_string_view<iter_value_t<_It>>;

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include "literal.h"
#include "test_iterators.h"

template <class It, class Sentinel, class CharT>
__host__ __device__ constexpr void test_ctad(cuda::std::basic_string_view<CharT> val)
{
  auto sv = cuda::std::basic_string_view(It(val.data()), Sentinel(It(val.data() + val.size())));
  static_assert(cuda::std::is_same_v<decltype(sv), cuda::std::basic_string_view<CharT>>);
  assert(sv.data() == val.data());
  assert(sv.size() == val.size());
}

template <class CharT>
__host__ __device__ constexpr void test_interator_deduct()
{
  cuda::std::basic_string_view<CharT> val{TEST_STRLIT(CharT, "test")};
  test_ctad<CharT*, CharT*>(val);
  test_ctad<CharT*, const CharT*>(val);
  test_ctad<const CharT*, CharT*>(val);
  test_ctad<const CharT*, sized_sentinel<const CharT*>>(val);
  test_ctad<contiguous_iterator<const CharT*>, contiguous_iterator<const CharT*>>(val);
  test_ctad<contiguous_iterator<const CharT*>, sized_sentinel<contiguous_iterator<const CharT*>>>(val);
}

__host__ __device__ constexpr bool test()
{
  test_interator_deduct<char>();
#if _CCCL_HAS_CHAR8_T()
  test_interator_deduct<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_interator_deduct<char16_t>();
  test_interator_deduct<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test_interator_deduct<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
