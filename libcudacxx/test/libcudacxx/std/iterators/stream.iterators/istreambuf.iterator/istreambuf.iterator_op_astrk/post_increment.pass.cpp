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

// proxy istreambuf_iterator<charT,traits>::operator++(int);

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::istringstream inf("abc");
    cuda::std::istreambuf_iterator<char> i(inf);
    assert(*i++ == 'a');
    assert(*i++ == 'b');
    assert(*i++ == 'c');
    assert(i == cuda::std::istreambuf_iterator<char>());
  }
  {
    cuda::std::wistringstream inf(L"abc");
    cuda::std::istreambuf_iterator<wchar_t> i(inf);
    assert(*i++ == L'a');
    assert(*i++ == L'b');
    assert(*i++ == L'c');
    assert(i == cuda::std::istreambuf_iterator<wchar_t>());
  }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
