//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// istreambuf_iterator

// template <class charT, class traits>
//   bool operator!=(const istreambuf_iterator<charT,traits>& a,
//                   const istreambuf_iterator<charT,traits>& b);

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::istringstream inf1("abc");
    cuda::std::istringstream inf2("def");
    cuda::std::istreambuf_iterator<char> i1(inf1);
    cuda::std::istreambuf_iterator<char> i2(inf2);
    cuda::std::istreambuf_iterator<char> i3;
    cuda::std::istreambuf_iterator<char> i4;
    cuda::std::istreambuf_iterator<char> i5(nullptr);

    assert(!(i1 != i1));
    assert(!(i1 != i2));
    assert((i1 != i3));
    assert((i1 != i4));
    assert((i1 != i5));

    assert(!(i2 != i1));
    assert(!(i2 != i2));
    assert((i2 != i3));
    assert((i2 != i4));
    assert((i2 != i5));

    assert((i3 != i1));
    assert((i3 != i2));
    assert(!(i3 != i3));
    assert(!(i3 != i4));
    assert(!(i3 != i5));

    assert((i4 != i1));
    assert((i4 != i2));
    assert(!(i4 != i3));
    assert(!(i4 != i4));
    assert(!(i4 != i5));

    assert((i5 != i1));
    assert((i5 != i2));
    assert(!(i5 != i3));
    assert(!(i5 != i4));
    assert(!(i5 != i5));
  }
  {
    cuda::std::wistringstream inf1(L"abc");
    cuda::std::wistringstream inf2(L"def");
    cuda::std::istreambuf_iterator<wchar_t> i1(inf1);
    cuda::std::istreambuf_iterator<wchar_t> i2(inf2);
    cuda::std::istreambuf_iterator<wchar_t> i3;
    cuda::std::istreambuf_iterator<wchar_t> i4;
    cuda::std::istreambuf_iterator<wchar_t> i5(nullptr);

    assert(!(i1 != i1));
    assert(!(i1 != i2));
    assert((i1 != i3));
    assert((i1 != i4));
    assert((i1 != i5));

    assert(!(i2 != i1));
    assert(!(i2 != i2));
    assert((i2 != i3));
    assert((i2 != i4));
    assert((i2 != i5));

    assert((i3 != i1));
    assert((i3 != i2));
    assert(!(i3 != i3));
    assert(!(i3 != i4));
    assert(!(i3 != i5));

    assert((i4 != i1));
    assert((i4 != i2));
    assert(!(i4 != i3));
    assert(!(i4 != i4));
    assert(!(i4 != i5));

    assert((i5 != i1));
    assert((i5 != i2));
    assert(!(i5 != i3));
    assert(!(i5 != i4));
    assert(!(i5 != i5));
  }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
