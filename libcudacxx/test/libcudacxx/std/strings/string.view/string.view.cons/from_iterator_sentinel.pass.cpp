//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

//  template <class It, class End>
//  constexpr basic_string_view(It begin, End end)

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/string_view>

#include "literal.h"
#include "test_iterators.h"

template <class It, class Sentinel, class CharT>
__host__ __device__ constexpr void test_from_iter_sentinel(cuda::std::basic_string_view<CharT> val)
{
  auto sv = cuda::std::basic_string_view<CharT>(It(val.data()), Sentinel(It(val.data() + val.size())));
  assert(sv.data() == val.data());
  assert(sv.size() == val.size());
}

template <class CharT>
__host__ __device__ constexpr void test_from_iter_sentinel()
{
  const cuda::std::basic_string_view<CharT> val = TEST_STRLIT(CharT, "test");
  test_from_iter_sentinel<CharT*, CharT*>(val);
  test_from_iter_sentinel<CharT*, const CharT*>(val);
  test_from_iter_sentinel<const CharT*, CharT*>(val);
  test_from_iter_sentinel<const CharT*, sized_sentinel<const CharT*>>(val);
  test_from_iter_sentinel<contiguous_iterator<const CharT*>, contiguous_iterator<const CharT*>>(val);
  test_from_iter_sentinel<contiguous_iterator<const CharT*>, sized_sentinel<contiguous_iterator<const CharT*>>>(val);

  using SV = cuda::std::basic_string_view<CharT>;

  static_assert(cuda::std::is_constructible_v<SV, const CharT*, CharT*>);
  static_assert(cuda::std::is_constructible_v<SV, CharT*, const CharT*>);
  static_assert(!cuda::std::is_constructible_v<SV, CharT*, void*>); // not a sentinel
  static_assert(!cuda::std::is_constructible_v<SV, signed char*, signed char*>); // wrong char type
  static_assert(!cuda::std::is_constructible_v<SV,
                                               random_access_iterator<CharT*>,
                                               random_access_iterator<CharT*>>); // not contiguous
  static_assert(cuda::std::is_constructible_v<SV, contiguous_iterator<CharT*>, contiguous_iterator<CharT*>>);
}

__host__ __device__ constexpr bool test()
{
  test_from_iter_sentinel<char>();
#if _CCCL_HAS_CHAR8_T()
  test_from_iter_sentinel<char8_t>();
#endif // _CCCL_HAS_CHAR8_T
  test_from_iter_sentinel<char16_t>();
  test_from_iter_sentinel<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test_from_iter_sentinel<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T

  return true;
}

#if _CCCL_HAS_EXCEPTIONS()
template <class CharT>
struct ThrowingSentinel
{
  friend bool operator==(const CharT*, const ThrowingSentinel&) noexcept
  {
    return true;
  }
  friend bool operator!=(const CharT*, const ThrowingSentinel&) noexcept
  {
    return false;
  }
  friend bool operator==(const ThrowingSentinel&, const CharT*) noexcept
  {
    return true;
  }
  friend bool operator!=(const ThrowingSentinel&, const CharT*) noexcept
  {
    return false;
  }
  friend cuda::std::iter_difference_t<const CharT*> operator-(const CharT*, ThrowingSentinel) noexcept
  {
    return {};
  }
  friend cuda::std::iter_difference_t<const CharT*> operator-(ThrowingSentinel, const CharT*)
  {
    throw 42;
  }
};
static_assert(cuda::std::sized_sentinel_for<ThrowingSentinel<char>, const char*>);

template <class CharT>
void test_from_iter_sentinel_exceptions()
{
  cuda::std::basic_string_view<CharT> val = TEST_STRLIT(CharT, "test");
  try
  {
    (void) cuda::std::basic_string_view<CharT>(val.data(), ThrowingSentinel<CharT>{});
    assert(false);
  }
  catch (int i)
  {
    assert(i == 42);
  }
  catch (...)
  {
    assert(false);
  }
}

void test_exceptions()
{
  test_from_iter_sentinel_exceptions<char>();
#  if _CCCL_HAS_CHAR8_T()
  test_from_iter_sentinel_exceptions<char8_t>();
#  endif // _CCCL_HAS_CHAR8_T
  test_from_iter_sentinel_exceptions<char16_t>();
  test_from_iter_sentinel_exceptions<char32_t>();
#  if _CCCL_HAS_WCHAR_T()
  test_from_iter_sentinel_exceptions<wchar_t>();
#  endif // _CCCL_HAS_WCHAR_T
}
#endif // _CCCL_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
  static_assert(test());
#if _CCCL_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // _CCCL_HAS_EXCEPTIONS()
  return 0;
}
