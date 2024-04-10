//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// template <class T, class charT = char, class traits = char_traits<charT>,
//           class Distance = ptrdiff_t>
// class ostream_iterator
//     : public iterator<output_iterator_tag, void, void, void, void>
// {
// public:
//     typedef charT char_type;
//     typedef traits traits_type;
//     typedef basic_istream<charT,traits> istream_type;
//     ...

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::ostream_iterator<double> I1;
#if TEST_STD_VER <= 2014
  static_assert(
    (cuda::std::is_convertible<I1, cuda::std::iterator<cuda::std::output_iterator_tag, void, void, void, void>>::value),
    "");
#else
  static_assert((cuda::std::is_same<I1::iterator_category, cuda::std::output_iterator_tag>::value), "");
  static_assert((cuda::std::is_same<I1::value_type, void>::value), "");
#  if TEST_STD_VER > 2017
  static_assert((cuda::std::is_same<I1::difference_type, ptrdiff_t>::value), "");
#  else
  static_assert((cuda::std::is_same<I1::difference_type, void>::value), "");
#  endif
  static_assert((cuda::std::is_same<I1::pointer, void>::value), "");
  static_assert((cuda::std::is_same<I1::reference, void>::value), "");
#endif
  static_assert((cuda::std::is_same<I1::char_type, char>::value), "");
  static_assert((cuda::std::is_same<I1::traits_type, cuda::std::char_traits<char>>::value), "");
  static_assert((cuda::std::is_same<I1::ostream_type, cuda::std::ostream>::value), "");
  typedef cuda::std::ostream_iterator<unsigned, wchar_t> I2;
#if TEST_STD_VER <= 2014
  static_assert(
    (cuda::std::is_convertible<I2, cuda::std::iterator<cuda::std::output_iterator_tag, void, void, void, void>>::value),
    "");
#else
  static_assert((cuda::std::is_same<I2::iterator_category, cuda::std::output_iterator_tag>::value), "");
  static_assert((cuda::std::is_same<I2::value_type, void>::value), "");
#  if TEST_STD_VER > 2017
  static_assert((cuda::std::is_same<I2::difference_type, ptrdiff_t>::value), "");
#  else
  static_assert((cuda::std::is_same<I2::difference_type, void>::value), "");
#  endif
  static_assert((cuda::std::is_same<I2::pointer, void>::value), "");
  static_assert((cuda::std::is_same<I2::reference, void>::value), "");
#endif
  static_assert((cuda::std::is_same<I2::char_type, wchar_t>::value), "");
  static_assert((cuda::std::is_same<I2::traits_type, cuda::std::char_traits<wchar_t>>::value), "");
  static_assert((cuda::std::is_same<I2::ostream_type, cuda::std::wostream>::value), "");

  return 0;
}
