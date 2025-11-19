//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization

// <random>

// class bernoulli_distribution

// template <class charT, class traits>
// basic_ostream<charT, traits>&
// operator<<(basic_ostream<charT, traits>& os,
//            const bernoulli_distribution& x);
//
// template <class charT, class traits>
// basic_istream<charT, traits>&
// operator>>(basic_istream<charT, traits>& is,
//            bernoulli_distribution& x);

#include <cuda/std/__random_>

#include "test_macros.h"

#if !_CCCL_COMPILER(NVRTC)
#  include <sstream>

void test()
{
  using D = cuda::std::bernoulli_distribution;
  D d1(.25);
  std::ostringstream os;
  os << d1;
  std::istringstream is(os.str());
  D d2;
  is >> d2;
  assert(d1 == d2);
}
#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, ({ test(); }));

  return 0;
}
