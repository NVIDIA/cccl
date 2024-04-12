//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// template<class charT, class traits = char_traits<charT> >
// class istreambuf_iterator
//     : public iterator<input_iterator_tag, charT,
//                       typename traits::off_type, unspecified,
//                       charT>
// {
// public:
//     typedef charT                         char_type;
//     typedef traits                        traits_type;
//     typedef typename traits::int_type     int_type;
//     typedef basic_streambuf<charT,traits> streambuf_type;
//     typedef basic_istream<charT,traits>   istream_type;
//     ...
//
// All specializations of istreambuf_iterator shall have a trivial copy constructor,
//    a constexpr default constructor and a trivial destructor.

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_STRING)
#  include <cuda/std/string>
#  include <cuda/std/type_traits>

#  include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::istreambuf_iterator<char> I1;
  static_assert((cuda::std::is_same<I1::iterator_category, cuda::std::input_iterator_tag>::value), "");
  static_assert((cuda::std::is_same<I1::value_type, char>::value), "");
  static_assert((cuda::std::is_same<I1::difference_type, cuda::std::char_traits<char>::off_type>::value), "");
  LIBCPP_STATIC_ASSERT((cuda::std::is_same<I1::pointer, char*>::value), "");
  static_assert((cuda::std::is_same<I1::reference, char>::value), "");
  static_assert((cuda::std::is_same<I1::char_type, char>::value), "");
  static_assert((cuda::std::is_same<I1::traits_type, cuda::std::char_traits<char>>::value), "");
  static_assert((cuda::std::is_same<I1::int_type, I1::traits_type::int_type>::value), "");
  static_assert((cuda::std::is_same<I1::streambuf_type, cuda::std::streambuf>::value), "");
  static_assert((cuda::std::is_same<I1::istream_type, cuda::std::istream>::value), "");
  static_assert((cuda::std::is_nothrow_default_constructible<I1>::value), "");
  static_assert((cuda::std::is_trivially_copy_constructible<I1>::value), "");
  static_assert((cuda::std::is_trivially_destructible<I1>::value), "");

  typedef cuda::std::istreambuf_iterator<wchar_t> I2;
  static_assert((cuda::std::is_same<I2::iterator_category, cuda::std::input_iterator_tag>::value), "");
  static_assert((cuda::std::is_same<I2::value_type, wchar_t>::value), "");
  static_assert((cuda::std::is_same<I2::difference_type, cuda::std::char_traits<wchar_t>::off_type>::value), "");
  LIBCPP_STATIC_ASSERT((cuda::std::is_same<I2::pointer, wchar_t*>::value), "");
  static_assert((cuda::std::is_same<I2::reference, wchar_t>::value), "");
  static_assert((cuda::std::is_same<I2::char_type, wchar_t>::value), "");
  static_assert((cuda::std::is_same<I2::traits_type, cuda::std::char_traits<wchar_t>>::value), "");
  static_assert((cuda::std::is_same<I2::int_type, I2::traits_type::int_type>::value), "");
  static_assert((cuda::std::is_same<I2::streambuf_type, cuda::std::wstreambuf>::value), "");
  static_assert((cuda::std::is_same<I2::istream_type, cuda::std::wistream>::value), "");
  static_assert((cuda::std::is_nothrow_default_constructible<I2>::value), "");
  static_assert((cuda::std::is_trivially_copy_constructible<I2>::value), "");
  static_assert((cuda::std::is_trivially_destructible<I2>::value), "");

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
