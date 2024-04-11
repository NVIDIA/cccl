//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// class ostream_iterator

// ostream_iterator& operator=(const T& value);

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

#  if defined(TEST_COMPILER_CLANG)
#    pragma clang diagnostic ignored "-Wliteral-conversion"
#  endif

#  ifdef TEST_COMPILER_MSVC
#    pragma warning(disable : 4244) // conversion from 'X' to 'Y', possible loss of data
#  endif

int main(int, char**)
{
  {
    cuda::std::ostringstream outf;
    cuda::std::ostream_iterator<int> i(outf);
    i = 2.4;
    assert(outf.str() == "2");
  }
  {
    cuda::std::ostringstream outf;
    cuda::std::ostream_iterator<int> i(outf, ", ");
    i = 2.4;
    assert(outf.str() == "2, ");
  }
  {
    cuda::std::wostringstream outf;
    cuda::std::ostream_iterator<int, wchar_t> i(outf);
    i = 2.4;
    assert(outf.str() == L"2");
  }
  {
    cuda::std::wostringstream outf;
    cuda::std::ostream_iterator<int, wchar_t> i(outf, L", ");
    i = 2.4;
    assert(outf.str() == L"2, ");
  }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
