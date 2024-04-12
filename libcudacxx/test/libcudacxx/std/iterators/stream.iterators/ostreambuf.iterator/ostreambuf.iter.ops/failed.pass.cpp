//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// class ostreambuf_iterator

// bool failed() const throw();

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

template <typename Char, typename Traits = cuda::std::char_traits<Char>>
struct my_streambuf : public cuda::std::basic_streambuf<Char, Traits>
{
  typedef typename cuda::std::basic_streambuf<Char, Traits>::int_type int_type;
  typedef typename cuda::std::basic_streambuf<Char, Traits>::char_type char_type;

  my_streambuf() {}
  int_type sputc(char_type)
  {
    return Traits::eof();
  }
};

int main(int, char**)
{
  {
    my_streambuf<char> buf;
    cuda::std::ostreambuf_iterator<char> i(&buf);
    i = 'a';
    assert(i.failed());
  }
  {
    my_streambuf<wchar_t> buf;
    cuda::std::ostreambuf_iterator<wchar_t> i(&buf);
    i = L'a';
    assert(i.failed());
  }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
