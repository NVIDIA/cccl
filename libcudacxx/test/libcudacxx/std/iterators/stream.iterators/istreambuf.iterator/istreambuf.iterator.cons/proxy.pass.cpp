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

// istreambuf_iterator(const proxy& p) throw();

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::istringstream inf("abc");
    cuda::std::istreambuf_iterator<char> j(inf);
    cuda::std::istreambuf_iterator<char> i = j++;
    assert(i != cuda::std::istreambuf_iterator<char>());
    assert(*i == 'b');
  }
  {
    cuda::std::wistringstream inf(L"abc");
    cuda::std::istreambuf_iterator<wchar_t> j(inf);
    cuda::std::istreambuf_iterator<wchar_t> i = j++;
    assert(i != cuda::std::istreambuf_iterator<wchar_t>());
    assert(*i == L'b');
  }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
