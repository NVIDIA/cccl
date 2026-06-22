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
#  include <complex>
#  include <tuple>
#  include <utility>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "MoveOnly.h"
#include "test_macros.h"

int main(int, char**)
{
  {
    using Pair = cuda::std::pair<MoveOnly, MoveOnly>;
    cuda::std::array<MoveOnly, 2> t0{MoveOnly(0), MoveOnly(1)};
    Pair p = cuda::std::move(t0);
    assert(cuda::std::get<0>(p) == 0);
    assert(cuda::std::get<1>(p) == 1);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 using Pair = cuda::std::pair<MoveOnly, MoveOnly>;
                 std::array<MoveOnly, 2> t0{MoveOnly(0), MoveOnly(1)};
                 Pair p = cuda::std::move(t0);
                 assert(cuda::std::get<0>(p) == 0);
                 assert(cuda::std::get<1>(p) == 1);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    using Pair = cuda::std::pair<MoveOnly, MoveOnly>;
    cuda::std::tuple<MoveOnly, MoveOnly> t0{MoveOnly(0), MoveOnly(1)};
    Pair p = cuda::std::move(t0);
    assert(cuda::std::get<0>(p) == 0);
    assert(cuda::std::get<1>(p) == 1);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 using Pair = cuda::std::pair<MoveOnly, MoveOnly>;
                 std::tuple<MoveOnly, MoveOnly> t0{MoveOnly(0), MoveOnly(1)};
                 Pair p = cuda::std::move(t0);
                 assert(cuda::std::get<0>(p) == 0);
                 assert(cuda::std::get<1>(p) == 1);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    using Pair = cuda::std::pair<float, double>;
    cuda::std::complex<float> t0{0.0f, 1.0};
    Pair p = cuda::std::move(t0);
    assert(cuda::std::get<0>(p) == 0.0f);
    assert(cuda::std::get<1>(p) == 1.0);
  }

#if _CCCL_HAS_HOST_STD_LIB() && defined(__cpp_lib_tuple_like)
  NV_IF_TARGET(NV_IS_HOST, ({
                 using Pair = cuda::std::pair<float, double>;
                 std::complex<float> t0{0.0f, 1.0};
                 Pair p = cuda::std::move(t0);
                 assert(cuda::std::get<0>(p) == 0.0f);
                 assert(cuda::std::get<1>(p) == 1.0);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && defined(__cpp_lib_tuple_like)

  return 0;
}
