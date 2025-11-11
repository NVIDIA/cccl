//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-localization

// <random>

// template<class RealType = double>
// class normal_distribution

// template <class CharT, class Traits, class RealType>
// basic_ostream<CharT, Traits>&
// operator<<(basic_ostream<CharT, Traits>& os,
//            const normal_distribution<RealType>& x);

// template <class CharT, class Traits, class RealType>
// basic_istream<CharT, Traits>&
// operator>>(basic_istream<CharT, Traits>& is,
//            normal_distribution<RealType>& x);

#include <cuda/std/__random_>
#include <cuda/std/cassert>

#include <sstream>

#include "test_macros.h"

#if !_CCCL_COMPILER(NVRTC)
void test_save_restore()
{
  using D = cuda::std::normal_distribution<>;
  D d1(7, 5);
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
  NV_IF_TARGET(NV_IS_HOST, ({ test_save_restore(); }));
  return 0;
}
