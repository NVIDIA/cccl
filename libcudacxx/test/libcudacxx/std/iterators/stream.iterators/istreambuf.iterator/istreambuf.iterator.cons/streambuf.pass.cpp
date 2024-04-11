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

// istreambuf_iterator(basic_streambuf<charT,traits>* s) throw();

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::istreambuf_iterator<char> i(nullptr);
    assert(i == cuda::std::istreambuf_iterator<char>());
  }
  {
    cuda::std::istringstream inf;
    cuda::std::istreambuf_iterator<char> i(inf.rdbuf());
    assert(i == cuda::std::istreambuf_iterator<char>());
  }
  {
    cuda::std::istringstream inf("a");
    cuda::std::istreambuf_iterator<char> i(inf.rdbuf());
    assert(i != cuda::std::istreambuf_iterator<char>());
  }
  {
    cuda::std::istreambuf_iterator<wchar_t> i(nullptr);
    assert(i == cuda::std::istreambuf_iterator<wchar_t>());
  }
  {
    cuda::std::wistringstream inf;
    cuda::std::istreambuf_iterator<wchar_t> i(inf.rdbuf());
    assert(i == cuda::std::istreambuf_iterator<wchar_t>());
  }
  {
    cuda::std::wistringstream inf(L"a");
    cuda::std::istreambuf_iterator<wchar_t> i(inf.rdbuf());
    assert(i != cuda::std::istreambuf_iterator<wchar_t>());
  }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
