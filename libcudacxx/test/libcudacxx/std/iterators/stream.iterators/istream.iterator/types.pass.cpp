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
// class istream_iterator
//     : public iterator<input_iterator_tag, T, Distance, const T*, const T&>
// {
// public:
//     typedef charT char_type;
//     typedef traits traits_type;
//     typedef basic_istream<charT,traits> istream_type;
//     ...
//
// Before C++17, we have:
//   If T is a literal type, then the default constructor shall be a constexpr constructor.
//   If T is a literal type, then this constructor shall be a trivial copy constructor.
//   If T is a literal type, then this destructor shall be a trivial destructor.
// C++17 says:
//   If is_trivially_default_constructible_v<T> is true, then
//       this constructor (the default ctor) is a constexpr constructor.
//   If is_trivially_copy_constructible_v<T> is true, then
//       this constructor (the copy ctor) is a trivial copy constructor.
//   If is_trivially_destructible_v<T> is true, then this
//       destructor is a trivial destructor.
//  Testing the C++17 ctors for this are in the ctor tests.

#include <cuda/std/iterator>
#include <cuda/std/type_traits>
#if defined(_LIBCUDACXX_HAS_STRING)
#  include <cuda/std/string>

#  include "test_macros.h"

int main(int, char**)
{
  typedef cuda::std::istream_iterator<double> I1; // double is trivially destructible
#  if TEST_STD_VER <= 2014
  static_assert(
    (cuda::std::is_convertible<
      I1,
      cuda::std::iterator<cuda::std::input_iterator_tag, double, cuda::std::ptrdiff_t, const double*, const double&>>::
       value),
    "");
#  else
  static_assert((cuda::std::is_same<I1::iterator_category, cuda::std::input_iterator_tag>::value), "");
  static_assert((cuda::std::is_same<I1::value_type, double>::value), "");
  static_assert((cuda::std::is_same<I1::difference_type, cuda::std::ptrdiff_t>::value), "");
  static_assert((cuda::std::is_same<I1::pointer, const double*>::value), "");
  static_assert((cuda::std::is_same<I1::reference, const double&>::value), "");
#  endif
  static_assert((cuda::std::is_same<I1::char_type, char>::value), "");
  static_assert((cuda::std::is_same<I1::traits_type, cuda::std::char_traits<char>>::value), "");
  static_assert((cuda::std::is_same<I1::istream_type, cuda::std::istream>::value), "");
  static_assert(cuda::std::is_trivially_copy_constructible<I1>::value, "");
  static_assert(cuda::std::is_trivially_destructible<I1>::value, "");

  typedef cuda::std::istream_iterator<unsigned, wchar_t> I2; // unsigned is trivially destructible
#  if TEST_STD_VER <= 2014
  static_assert(
    (cuda::std::is_convertible<
      I2,
      cuda::std::
        iterator<cuda::std::input_iterator_tag, unsigned, cuda::std::ptrdiff_t, const unsigned*, const unsigned&>>::
       value),
    "");
#  else
  static_assert((cuda::std::is_same<I2::iterator_category, cuda::std::input_iterator_tag>::value), "");
  static_assert((cuda::std::is_same<I2::value_type, unsigned>::value), "");
  static_assert((cuda::std::is_same<I2::difference_type, cuda::std::ptrdiff_t>::value), "");
  static_assert((cuda::std::is_same<I2::pointer, const unsigned*>::value), "");
  static_assert((cuda::std::is_same<I2::reference, const unsigned&>::value), "");
#  endif
  static_assert((cuda::std::is_same<I2::char_type, wchar_t>::value), "");
  static_assert((cuda::std::is_same<I2::traits_type, cuda::std::char_traits<wchar_t>>::value), "");
  static_assert((cuda::std::is_same<I2::istream_type, cuda::std::wistream>::value), "");
  static_assert(cuda::std::is_trivially_copy_constructible<I2>::value, "");
  static_assert(cuda::std::is_trivially_destructible<I2>::value, "");

  typedef cuda::std::istream_iterator<cuda::std::string> I3; // string is NOT trivially destructible
  static_assert(!cuda::std::is_trivially_copy_constructible<I3>::value, "");
  static_assert(!cuda::std::is_trivially_destructible<I3>::value, "");

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
