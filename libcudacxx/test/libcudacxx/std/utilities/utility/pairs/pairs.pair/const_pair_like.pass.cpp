//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/utility>

// template <class T1, class T2> struct pair

// template <pair-like Pair> EXPLICIT constexpr pair(Pair&& p);

#include <cuda/std/__memory_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <array>
#  include <utility>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "test_macros.h"

struct Empty
{};

int main(int, char**)
{
  {
    using Pair = cuda::std::pair<int, short>;
    const cuda::std::array<long, 2> t0{1337, 42};
    Pair p2 = t0;
    assert(cuda::std::get<0>(p2) == 1337);
    assert(cuda::std::get<1>(p2) == 42);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 using Pair = cuda::std::pair<int, short>;
                 const std::array<long, 2> t0{1337, 42};
                 Pair p2 = t0;
                 assert(cuda::std::get<0>(p2) == 1337);
                 assert(cuda::std::get<1>(p2) == 42);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    using Pair = cuda::std::pair<int, short>;
    const cuda::std::tuple<long, long> t0{1337, 42};
    Pair p2 = t0;
    assert(cuda::std::get<0>(p2) == 1337);
    assert(cuda::std::get<1>(p2) == 42);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 using Pair = cuda::std::pair<int, short>;
                 const std::tuple<long, long> t0{1337, 42};
                 Pair p2 = t0;
                 assert(cuda::std::get<0>(p2) == 1337);
                 assert(cuda::std::get<1>(p2) == 42);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    using Pair = cuda::std::pair<float, double>;
    const cuda::std::complex<double> t0{0.0, 1.0};
    Pair p2 = t0;
    assert(cuda::std::get<0>(p2) == 0.0f);
    assert(cuda::std::get<1>(p2) == 1.0);
  }

#if _CCCL_HAS_HOST_STD_LIB() && defined(__cpp_lib_tuple_like)
  NV_IF_TARGET(NV_IS_HOST, ({
                 using Pair = cuda::std::pair<float, double>;
                 const cuda::std::complex<double> t0{0.0, 1.0};
                 Pair p2 = t0;
                 assert(cuda::std::get<0>(p2) == 0.0f);
                 assert(cuda::std::get<1>(p2) == 1.0);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && defined(__cpp_lib_tuple_like)

  return 0;
}
