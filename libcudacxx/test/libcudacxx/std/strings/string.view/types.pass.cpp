//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/string_view>

// Test nested types and default template args:

// template<class charT, class traits = char_traits<charT>>
// {
// public:
//     // types:
//     using traits_type               = traits;
//     using value_type                = charT;
//     using pointer                   = value_type*;
//     using const_pointer             = const value_type*;
//     using reference                 = value_type&;
//     using const_reference           = const value_type&;
//     using const_iterator            = implementation-defined ; // see 24.4.2.2
//     using iterator                  = const_iterator;
//     using const_reverse_iterator    = reverse_iterator<const_iterator>;
//     using iterator                  = const_reverse_iterator;
//     using size_type                 = size_t;
//     using difference_type           = ptrdiff_t;
//     static constexpr size_type npos = size_type(-1);
//
// };

#include <cuda/std/cstddef>
#include <cuda/std/iterator>
#include <cuda/std/string_view>
#include <cuda/std/type_traits>

template <class Traits>
__host__ __device__ constexpr void test_type()
{
  using SV = cuda::std::basic_string_view<typename Traits::char_type, Traits>;

  static_assert(cuda::std::is_same_v<typename SV::traits_type, Traits>);
  static_assert(cuda::std::is_same_v<typename SV::value_type, typename Traits::char_type>);
  static_assert(cuda::std::is_same_v<typename SV::size_type, cuda::std::size_t>);
  static_assert(cuda::std::is_same_v<typename SV::difference_type, cuda::std::ptrdiff_t>);
  static_assert(cuda::std::is_same_v<typename SV::reference, typename SV::value_type&>);
  static_assert(cuda::std::is_same_v<typename SV::const_reference, const typename SV::value_type&>);
  static_assert(cuda::std::is_same_v<typename SV::pointer, typename SV::value_type*>);
  static_assert(cuda::std::is_same_v<typename SV::const_pointer, const typename SV::value_type*>);
  static_assert(cuda::std::is_same_v<typename cuda::std::iterator_traits<typename SV::iterator>::iterator_category,
                                     cuda::std::random_access_iterator_tag>);
  static_assert(
    cuda::std::is_same_v<typename cuda::std::iterator_traits<typename SV::const_iterator>::iterator_category,
                         cuda::std::random_access_iterator_tag>);
  static_assert(
    cuda::std::is_same_v<typename SV::reverse_iterator, cuda::std::reverse_iterator<typename SV::iterator>>);
  static_assert(
    cuda::std::is_same_v<typename SV::const_reverse_iterator, cuda::std::reverse_iterator<typename SV::const_iterator>>);
  static_assert(SV::npos == static_cast<typename SV::size_type>(-1));
  static_assert(cuda::std::is_same_v<typename SV::iterator, typename SV::const_iterator>);
  static_assert(cuda::std::is_same_v<typename SV::reverse_iterator, typename SV::const_reverse_iterator>);
}

__host__ __device__ constexpr bool test()
{
  test_type<cuda::std::char_traits<char>>();
#if _CCCL_HAS_CHAR8_T()
  test_type<cuda::std::char_traits<char8_t>>();
#endif // _CCCL_HAS_CHAR8_T()
  test_type<cuda::std::char_traits<char16_t>>();
  test_type<cuda::std::char_traits<char32_t>>();
#if _CCCL_HAS_WCHAR_T()
  test_type<cuda::std::char_traits<wchar_t>>();
#endif // _CCCL_HAS_WCHAR_T()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
