//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include <string>
#include <string_view>

#include "literal.h"

#if __cpp_lib_string_view >= 201606L

template <class CharT>
constexpr void test_with_default_type_traits()
{
  using HostS  = std::basic_string<CharT>;
  using HostSV = std::basic_string_view<CharT>;
  using CudaSV = cuda::std::basic_string_view<CharT>;

  // check that cuda::std::char_traits are mapped to std::char_traits
  static_assert(cuda::std::is_same_v<typename HostS::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename HostS::traits_type, std::char_traits<CharT>>);
  static_assert(cuda::std::is_same_v<typename CudaSV::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename CudaSV::traits_type, cuda::std::char_traits<CharT>>);

  const CharT* str = TEST_STRLIT(CharT, "some text");

  // test conversion to std::basic_string_view<CharT, std::char_traits<CharT>>
  {
    static_assert(cuda::std::is_convertible_v<CudaSV, HostS> == cuda::std::is_convertible_v<HostSV, HostS>);
    static_assert(!noexcept(HostS{CudaSV{}}));

    CudaSV cuda_sv{str};
    HostS host_s{cuda_sv};
    assert(host_s == cuda_sv);
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
  using HostSV = std::basic_string_view<CharT, custom_type_traits<CharT>>;
  using CudaSV = cuda::std::basic_string_view<CharT, custom_type_traits<CharT>>;

  // check that cuda::std::char_traits are mapped to std::char_traits
  static_assert(cuda::std::is_same_v<typename HostS::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename HostS::traits_type, custom_type_traits<CharT>>);
  static_assert(cuda::std::is_same_v<typename CudaSV::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename CudaSV::traits_type, custom_type_traits<CharT>>);

  const CharT* str = TEST_STRLIT(CharT, "some text");

  // test conversion to std::basic_string_view<CharT, custom_type_traits<CharT>>
  {
    static_assert(cuda::std::is_convertible_v<CudaSV, HostS> == cuda::std::is_convertible_v<HostSV, HostS>);
    static_assert(!noexcept(HostS{CudaSV{}}));

    CudaSV cuda_sv{str};
    HostS host_s{cuda_sv};
    assert(host_s == cuda_sv);
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
#  if _CCCL_HAS_CHAR8_T()
  test_type<char8_t>();
#  endif // _CCCL_HAS_CHAR8_T()
  test_type<char16_t>();
  test_type<char32_t>();
#  if _CCCL_HAS_WCHAR_T()
  test_type<wchar_t>();
#  endif // _CCCL_HAS_WCHAR_T()

  return true;
}

// clang fails due to assignment to member of a union with no active member inside libstdc++
// gcc-12 + nvcc 12.0 warns about accessing expired storage
#  if __cpp_lib_constexpr_string >= 201907L && !_CCCL_COMPILER(CLANG) \
    && !(_CCCL_COMPILER(GCC, ==, 12) && _CCCL_CUDA_COMPILER(NVCC, ==, 12, 0))
static_assert(test());
#  endif // __cpp_lib_constexpr_string >= 201907L && !_CCCL_COMPILER(CLANG) && !(_CCCL_COMPILER(GCC, ==, 12) &&
         // _CCCL_CUDA_COMPILER(NVCC, ==, 12, 0))

#endif // __cpp_lib_string_view >= 201606L

int main(int, char**)
{
#if __cpp_lib_string_view >= 201606L
  NV_IF_TARGET(NV_IS_HOST, (test();))
#endif // __cpp_lib_string_view >= 201606L
  return 0;
}
