//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// <cuda/std/string_view>

// template<class CharT, class Traits, class Allocator>
//   constexpr basic_string<CharT, Traits, Allocator>
//   operator+(const basic_string<CharT, Traits, Allocator>& lhs,
//             type_identity_t<cuda::std::basic_string_view<CharT, Traits>> rhs);
//
// template<class CharT, class Traits, class Allocator>
//   constexpr basic_string<CharT, Traits, Allocator>
//   operator+(basic_string<CharT, Traits, Allocator>&& lhs,
//             type_identity_t<cuda::std::basic_string_view<CharT, Traits>> rhs);
//
// template<class CharT, class Traits, class Allocator>
//   constexpr basic_string<CharT, Traits, Allocator>
//   operator+(type_identity_t<cuda::std::basic_string_view<CharT, Traits>> lhs,
//             const basic_string<CharT, Traits, Allocator>& rhs);
//
// template<class CharT, class Traits, class Allocator>
//   constexpr basic_string<CharT, Traits, Allocator>
//   operator+(type_identity_t<cuda::std::basic_string_view<CharT, Traits>> lhs,
//             basic_string<CharT, Traits, Allocator>&& rhs);

#include <cuda/std/cassert>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/utility>

#include <string>

#include "literal.h"

template <class CharT>
struct CustomCharTraits : cuda::std::char_traits<CharT>
{};

template <class HostS, class CudaSV>
constexpr void test_string_view_as_rhs_case(
  const typename HostS::value_type* lhs_chars,
  const typename HostS::value_type* rhs_chars,
  const typename HostS::value_type* expected_chars)
{
  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::declval<const HostS&>() + cuda::std::declval<CudaSV>()), HostS>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<HostS&&>() + cuda::std::declval<CudaSV>()), HostS>);

  HostS expected{expected_chars};

  {
    HostS lhs{lhs_chars};
    CudaSV rhs{rhs_chars};
    HostS result = lhs + rhs;
    assert(result == expected);
    assert(lhs == HostS{lhs_chars});
  }
  {
    HostS lhs{lhs_chars};
    CudaSV rhs{rhs_chars};
    HostS result = cuda::std::move(lhs) + rhs;
    assert(result == expected);
  }
}

template <class HostS, class CudaSV>
constexpr void test_string_view_as_lhs_case(
  const typename HostS::value_type* lhs_chars,
  const typename HostS::value_type* rhs_chars,
  const typename HostS::value_type* expected_chars)
{
  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::declval<CudaSV>() + cuda::std::declval<const HostS&>()), HostS>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<CudaSV>() + cuda::std::declval<HostS&&>()), HostS>);

  HostS expected{expected_chars};

  {
    CudaSV lhs{lhs_chars};
    HostS rhs{rhs_chars};
    HostS result = lhs + rhs;
    assert(result == expected);
    assert(rhs == HostS{rhs_chars});
  }
  {
    CudaSV lhs{lhs_chars};
    HostS rhs{rhs_chars};
    HostS result = lhs + cuda::std::move(rhs);
    assert(result == expected);
  }
}

template <class HostS, class CudaSV>
constexpr void test_combination()
{
  using CharT = typename HostS::value_type;

  test_string_view_as_rhs_case<HostS, CudaSV>(
    TEST_STRLIT(CharT, "left"), TEST_STRLIT(CharT, "right"), TEST_STRLIT(CharT, "leftright"));
  test_string_view_as_rhs_case<HostS, CudaSV>(
    TEST_STRLIT(CharT, ""), TEST_STRLIT(CharT, "right"), TEST_STRLIT(CharT, "right"));
  test_string_view_as_rhs_case<HostS, CudaSV>(
    TEST_STRLIT(CharT, "left"), TEST_STRLIT(CharT, ""), TEST_STRLIT(CharT, "left"));

  test_string_view_as_lhs_case<HostS, CudaSV>(
    TEST_STRLIT(CharT, "left"), TEST_STRLIT(CharT, "right"), TEST_STRLIT(CharT, "leftright"));
  test_string_view_as_lhs_case<HostS, CudaSV>(
    TEST_STRLIT(CharT, ""), TEST_STRLIT(CharT, "right"), TEST_STRLIT(CharT, "right"));
  test_string_view_as_lhs_case<HostS, CudaSV>(
    TEST_STRLIT(CharT, "left"), TEST_STRLIT(CharT, ""), TEST_STRLIT(CharT, "left"));
}

template <class CharT>
constexpr void test_with_default_type_traits()
{
  using HostS  = std::basic_string<CharT>;
  using CudaSV = cuda::std::basic_string_view<CharT>;

  static_assert(cuda::std::is_same_v<typename HostS::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename HostS::traits_type, std::char_traits<CharT>>);
  static_assert(cuda::std::is_same_v<typename CudaSV::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename CudaSV::traits_type, cuda::std::char_traits<CharT>>);

  test_combination<HostS, CudaSV>();
}

template <class CharT>
constexpr void test_with_custom_type_traits()
{
  using Traits = CustomCharTraits<CharT>;
  using HostS  = std::basic_string<CharT, Traits>;
  using CudaSV = cuda::std::basic_string_view<CharT, Traits>;

  static_assert(cuda::std::is_same_v<typename HostS::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename HostS::traits_type, Traits>);
  static_assert(cuda::std::is_same_v<typename CudaSV::value_type, CharT>);
  static_assert(cuda::std::is_same_v<typename CudaSV::traits_type, Traits>);

  test_combination<HostS, CudaSV>();
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

// clang fails due to assignment to member of a union with no active member inside libstdc++
// gcc-12 + nvcc 12.0 warns about accessing expired storage
#if __cpp_lib_constexpr_string >= 201907L && !_CCCL_COMPILER(CLANG) \
  && !(_CCCL_COMPILER(GCC, ==, 12) && _CCCL_CUDA_COMPILER(NVCC, ==, 12, 0))
static_assert(test());
#endif // __cpp_lib_constexpr_string >= 201907L && !_CCCL_COMPILER(CLANG) && !(_CCCL_COMPILER(GCC, ==, 12) &&
       // _CCCL_CUDA_COMPILER(NVCC, ==, 12, 0))

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
