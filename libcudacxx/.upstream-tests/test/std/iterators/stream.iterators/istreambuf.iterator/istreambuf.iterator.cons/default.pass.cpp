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
//
// istreambuf_iterator() throw();
//
// All specializations of istreambuf_iterator shall have a trivial copy constructor,
//    a constexpr default constructor and a trivial destructor.

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_SSTREAM)
#include <cuda/std/sstream>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef cuda::std::istreambuf_iterator<char> T;
        T it;
        assert(it == T());
#if TEST_STD_VER >= 11
        constexpr T it2;
        (void)it2;
#endif
    }
    {
        typedef cuda::std::istreambuf_iterator<wchar_t> T;
        T it;
        assert(it == T());
#if TEST_STD_VER >= 11
        constexpr T it2;
        (void)it2;
#endif
    }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
