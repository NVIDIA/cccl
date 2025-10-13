//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// constexpr basic_string_view(std::basic_string sv);

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/utility>

#include <string>

#include "literal.h"

template <class CharT>
constexpr void test_with_default_type_traits()
{
  using HostS  = std::basic_string<CharT>;
  using CudaSV = cuda::std::basic_string_view<CharT>;

  // check that cuda::std::char_traits are mapped to std::char_traits
  static_assert(cuda::std::is_same_v<typename HostS::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename HostS::traits_type, std::char_traits<CharT>>);
  static_assert(cuda::std::is_same_v<typename CudaSV::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename CudaSV::traits_type, cuda::std::char_traits<CharT>>);

  const CharT* str = TEST_STRLIT(CharT, "some text");

  // test construction from std::basic_string<CharT, std::char_traits<CharT>>
  {
    static_assert(cuda::std::is_constructible_v<CudaSV, HostS>);
    static_assert(noexcept(CudaSV{cuda::std::declval<HostS>()}));

    HostS host_s{str};
    CudaSV cuda_sv{host_s};
    assert(cuda_sv.data() == host_s.data());
    assert(cuda_sv.size() == host_s.size());
  }
}

template <class CharT>
struct custom_type_traits
    : private std::char_traits<CharT>
    , private cuda::std::char_traits<CharT>
{
  using char_type  = typename cuda::std::char_traits<CharT>::char_type;
  using int_type   = typename cuda::std::char_traits<CharT>::int_type;
  using pos_type   = typename std::char_traits<CharT>::pos_type;
  using off_type   = typename std::char_traits<CharT>::off_type;
  using state_type = typename std::char_traits<CharT>::state_type;

  using cuda::std::char_traits<CharT>::assign;
  using cuda::std::char_traits<CharT>::eq;
  using cuda::std::char_traits<CharT>::lt;
  using cuda::std::char_traits<CharT>::compare;
  using cuda::std::char_traits<CharT>::length;
  using cuda::std::char_traits<CharT>::find;
  using cuda::std::char_traits<CharT>::move;
  using cuda::std::char_traits<CharT>::copy;
  using cuda::std::char_traits<CharT>::to_char_type;
  using cuda::std::char_traits<CharT>::to_int_type;
  using cuda::std::char_traits<CharT>::eq_int_type;
  using std::char_traits<CharT>::eof;
  using std::char_traits<CharT>::not_eof;
};

template <class CharT>
constexpr void test_with_custom_type_traits()
{
  using HostS  = std::basic_string<CharT, custom_type_traits<CharT>>;
  using CudaSV = cuda::std::basic_string_view<CharT, custom_type_traits<CharT>>;

  // check that cuda::std::char_traits are mapped to std::char_traits
  static_assert(cuda::std::is_same_v<typename HostS::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename HostS::traits_type, custom_type_traits<CharT>>);
  static_assert(cuda::std::is_same_v<typename CudaSV::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename CudaSV::traits_type, custom_type_traits<CharT>>);

  const CharT* str = TEST_STRLIT(CharT, "some text");

  // test construction from std::basic_string_view<CharT, custom_type_traits<CharT>>
  {
    static_assert(cuda::std::is_constructible_v<CudaSV, HostS>);
    static_assert(noexcept(CudaSV{cuda::std::declval<HostS>()}));

    HostS host_s{str};
    CudaSV cuda_sv{host_s};
    assert(cuda_sv.data() == host_s.data());
    assert(cuda_sv.size() == host_s.size());
  }
}

template <class CharT>
constexpr void test_type()
{
  test_with_default_type_traits<CharT>();
  test_with_custom_type_traits<CharT>();
}

constexpr bool test()
{
  test_type<char>();
#if _CCCL_HAS_CHAR8_T()
  test_type<char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_type<char16_t>();
  test_type<char32_t>();
#if _CCCL_HAS_WCHAR_T()
  test_type<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

#if __cpp_lib_constexpr_string >= 201907L
static_assert(test());
#endif // __cpp_lib_constexpr_string >= 201907L

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
