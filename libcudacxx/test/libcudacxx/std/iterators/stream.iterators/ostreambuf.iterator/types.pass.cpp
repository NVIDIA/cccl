//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// template <class charT, class traits = char_traits<charT> >
// class ostreambuf_iterator
//   : public iterator<output_iterator_tag, void, void, void, void>
// {
// public:
//   typedef charT                          char_type;
//   typedef traits                         traits_type;
//   typedef basic_streambuf<charT, traits> streambuf_type;
//   typedef basic_ostream<charT, traits>   ostream_type;
//   ...

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_STRING)
#  include <cuda/std/string>
#  include <cuda/std/type_traits>

#  include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::ostreambuf_iterator<char> I1;

  static_assert((cuda::std::is_same<I1::iterator_category, cuda::std::output_iterator_tag>::value), "");
  static_assert((cuda::std::is_same<I1::value_type, void>::value), "");
  static_assert((cuda::std::is_same<I1::difference_type, void>::value), "");
  static_assert((cuda::std::is_same<I1::pointer, void>::value), "");
  static_assert((cuda::std::is_same<I1::reference, void>::value), "");
  static_assert((cuda::std::is_same<I1::char_type, char>::value), "");
  static_assert((cuda::std::is_same<I1::traits_type, cuda::std::char_traits<char>>::value), "");
  static_assert((cuda::std::is_same<I1::streambuf_type, cuda::std::streambuf>::value), "");
  static_assert((cuda::std::is_same<I1::ostream_type, cuda::std::ostream>::value), "");

  typedef cuda::std::ostreambuf_iterator<wchar_t> I2;

  static_assert((cuda::std::is_same<I2::iterator_category, cuda::std::output_iterator_tag>::value), "");
  static_assert((cuda::std::is_same<I2::value_type, void>::value), "");
  static_assert((cuda::std::is_same<I2::difference_type, void>::value), "");
  static_assert((cuda::std::is_same<I2::pointer, void>::value), "");
  static_assert((cuda::std::is_same<I2::reference, void>::value), "");
  static_assert((cuda::std::is_same<I2::char_type, wchar_t>::value), "");
  static_assert((cuda::std::is_same<I2::traits_type, cuda::std::char_traits<wchar_t>>::value), "");
  static_assert((cuda::std::is_same<I2::streambuf_type, cuda::std::wstreambuf>::value), "");
  static_assert((cuda::std::is_same<I2::ostream_type, cuda::std::wostream>::value), "");

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
