//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization
// XFAIL: true

// <random>

// template<class _IntType = int>
// class uniform_int_distribution

// template <class charT, class traits>
// basic_ostream<charT, traits>&
// operator<<(basic_ostream<charT, traits>& os,
//            const uniform_int_distribution& x);
//
// template <class charT, class traits>
// basic_istream<charT, traits>&
// operator>>(basic_istream<charT, traits>& is,
//            uniform_int_distribution& x);

#include <cuda/std/__random_>
#include <cuda/std/cassert>
#include <cuda/std/sstream>

#include "test_macros.h"

int main(int, char**)
{
  {
    using D = cuda::std::uniform_int_distribution<>;
    D d1(3, 8);
    cuda::std::ostringstream os;
    os << d1;
    cuda::std::istringstream is(os.str());
    D d2;
    is >> d2;
    assert(d1 == d2);
  }

  return 0;
}
