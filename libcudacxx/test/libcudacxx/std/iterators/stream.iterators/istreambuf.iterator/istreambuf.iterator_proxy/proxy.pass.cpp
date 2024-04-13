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
//                       typename traits::off_type, charT*,
//                       charT>
// {
// public:
//     ...
//     proxy operator++(int);

// class proxy
// {
// public:
//     charT operator*();
// };

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
  }
  {
    cuda::std::wistringstream inf(L"abc");
    cuda::std::istreambuf_iterator<wchar_t> i(inf);
    assert(*i++ == L'a');
  }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
