//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template<class Tuple>
// tuple(Tuple&& u) = default;
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

#include "test_macros.h"

struct Empty
{};

int main(int, char**)
{
  {
    using T = cuda::std::tuple<Empty>;
    const cuda::std::array<Empty, 1> t0{{}};
    [[maybe_unused]] T t = t0;
  }

  {
    using T = cuda::std::tuple<int>;
    const cuda::std::array<long, 1> t0{42};
    T t = t0;
    assert(cuda::std::get<0>(t) == 42);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 using T = cuda::std::tuple<int>;
                 const std::array<long, 1> t0{42};
                 T t = t0;
                 assert(cuda::std::get<0>(t) == 42);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    using T = cuda::std::tuple<int, short>;
    const cuda::std::array<long, 2> t0{1337, 42};
    T t = t0;
    assert(cuda::std::get<0>(t) == 1337);
    assert(cuda::std::get<1>(t) == 42);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 using T = cuda::std::tuple<int, short>;
                 const std::array<long, 2> t0{1337, 42};
                 T t = t0;
                 assert(cuda::std::get<0>(t) == 1337);
                 assert(cuda::std::get<1>(t) == 42);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    using T = cuda::std::tuple<int, short>;
    const cuda::std::pair<long, long> t0{1337, 42};
    T t = t0;
    assert(cuda::std::get<0>(t) == 1337);
    assert(cuda::std::get<1>(t) == 42);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 using T = cuda::std::tuple<int, short>;
                 const std::pair<long, long> t0{1337, 42};
                 T t = t0;
                 assert(cuda::std::get<0>(t) == 1337);
                 assert(cuda::std::get<1>(t) == 42);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    using T = cuda::std::tuple<float, double>;
    const cuda::std::complex<double> t0{0.0, 1.0};
    T t = t0;
    assert(cuda::std::get<0>(t) == 0.0f);
    assert(cuda::std::get<1>(t) == 1.0);
  }

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(NV_IS_HOST, ({
                 using T = cuda::std::tuple<float, double>;
                 const std::complex<double> t0{0.0, 1.0};
                 T t = t0;
                 assert(cuda::std::get<0>(t) == 0.0f);
                 assert(cuda::std::get<1>(t) == 1.0);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L

  {
    using T = cuda::std::tuple<int, long, double>;
    const cuda::std::array<int, 3> t0{42, 1337, 0};
    T t = t0;
    assert(cuda::std::get<0>(t) == 42);
    assert(cuda::std::get<1>(t) == 1337);
    assert(cuda::std::get<2>(t) == 0.0);
  }

  return 0;
}
