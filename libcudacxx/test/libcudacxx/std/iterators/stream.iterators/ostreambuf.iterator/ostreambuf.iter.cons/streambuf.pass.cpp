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

// ostreambuf_iterator(streambuf_type* s) throw();

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#  include <cuda/std/cassert>
#  include <cuda/std/sstream>

#  include "test_macros.h"

int main(int, char**)
{
  {
    cuda::std::ostringstream outf;
    cuda::std::ostreambuf_iterator<char> i(outf.rdbuf());
    assert(!i.failed());
  }
  {
    cuda::std::wostringstream outf;
    cuda::std::ostreambuf_iterator<wchar_t> i(outf.rdbuf());
    assert(!i.failed());
  }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
