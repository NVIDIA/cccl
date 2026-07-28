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

#include "copy_move_types.h"
#include "test_macros.h"

TEST_FUNC constexpr bool test()
{
  // test implicit conversions.
  {
    using Pair = cuda::std::pair<ConvertibleFrom<int>, ConvertibleFrom<int>>;
    const cuda::std::tuple<int, int> p{1, 2};
    Pair p2 = p;
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 2);
    static_assert(cuda::std::is_nothrow_constructible_v<Pair, const cuda::std::tuple<int, int>&>);
  }

  { // Ensure we properly detect noexcept
    using Pair = cuda::std::pair<ConvertibleFrom<int, false>, ConvertibleFrom<int>>;
    const cuda::std::tuple<int, int> p{1, 2};
    Pair p2 = p;
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 2);
    static_assert(!cuda::std::is_nothrow_constructible_v<Pair, const cuda::std::tuple<int, int>&>);
  }

  { // Ensure we properly detect noexcept
    using Pair = cuda::std::pair<ConvertibleFrom<int>, ConvertibleFrom<int, false>>;
    const cuda::std::tuple<int, int> p{1, 2};
    Pair p2 = p;
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 2);
    static_assert(!cuda::std::is_nothrow_constructible_v<Pair, const cuda::std::tuple<int, int>&>);
  }

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(NV_IS_HOST, ({
                 const std::tuple<int, int> p{1, 2};
                 cuda::std::pair<ConvertibleFrom<int>, ConvertibleFrom<int>> p2 = p;
                 assert(cuda::std::get<0>(p2).v == 1);
                 assert(cuda::std::get<1>(p2).v == 2);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)

  {
    const cuda::std::array<int, 2> p{1, 42};
    cuda::std::pair<ConvertibleFrom<int>, ConvertibleFrom<int>> p2 = p;
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 42);
  }

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(NV_IS_HOST, ({
                 const std::array<int, 2> p{1, 42};
                 cuda::std::pair<ConvertibleFrom<int>, ConvertibleFrom<int>> p2 = p;
                 assert(cuda::std::get<0>(p2).v == 1);
                 assert(cuda::std::get<1>(p2).v == 42);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)

  {
    const cuda::std::complex<float> p{1.0f, 42.0f};
    cuda::std::pair<ConvertibleFrom<float>, ConvertibleFrom<float>> p2 = p;
    assert(cuda::std::get<0>(p2).v == 1.0f);
    assert(cuda::std::get<1>(p2).v == 42.0f);
  }

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(NV_IS_HOST, ({
                 const std::complex<float> p{1.0f, 42.0f};
                 cuda::std::pair<ConvertibleFrom<float>, ConvertibleFrom<float>> p2 = p;
                 assert(cuda::std::get<0>(p2).v == 1.0f);
                 assert(cuda::std::get<1>(p2).v == 42.0f);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L

  // test explicit conversions.
  {
    using Pair = cuda::std::pair<ConvertibleFrom<int>, ConvertibleFrom<int>>;
    const cuda::std::pair<int, int> p{1, 2};
    Pair p2{p};
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 2);
    static_assert(cuda::std::is_nothrow_constructible_v<Pair, const cuda::std::tuple<int, int>&>);
  }

  { // Ensure we properly detect noexcept
    using Pair = cuda::std::pair<ConvertibleFrom<int, false>, ConvertibleFrom<int>>;
    const cuda::std::tuple<int, int> p{1, 2};
    Pair p2{p};
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 2);
    static_assert(!cuda::std::is_nothrow_constructible_v<Pair, const cuda::std::tuple<int, int>&>);
  }

  { // Ensure we properly detect noexcept
    using Pair = cuda::std::pair<ConvertibleFrom<int>, ConvertibleFrom<int, false>>;
    const cuda::std::tuple<int, int> p{1, 2};
    Pair p2{p};
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 2);
    static_assert(!cuda::std::is_nothrow_constructible_v<Pair, const cuda::std::tuple<int, int>&>);
  }

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(NV_IS_HOST, ({
                 const std::pair<int, int> p{1, 2};
                 cuda::std::pair<ExplicitConstructibleFrom<int>, ExplicitConstructibleFrom<int>> p2{p};
                 assert(cuda::std::get<0>(p2).v == 1);
                 assert(cuda::std::get<1>(p2).v == 2);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)

  {
    using Pair = cuda::std::pair<ExplicitConstructibleFrom<int>, ConvertibleFrom<int>>;
    const cuda::std::array<int, 2> p{1, 42};
    Pair p2{p};
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 42);
  }

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(NV_IS_HOST, ({
                 const std::array<int, 2> p{1, 42};
                 cuda::std::pair<ExplicitConstructibleFrom<int>, ConvertibleFrom<int>> p2{p};
                 assert(cuda::std::get<0>(p2).v == 1);
                 assert(cuda::std::get<1>(p2).v == 42);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)

  {
    const cuda::std::complex<float> p{1.0f, 42.0f};
    cuda::std::pair<ExplicitConstructibleFrom<float>, ConvertibleFrom<float>> p2{p};
    assert(cuda::std::get<0>(p2).v == 1.0f);
    assert(cuda::std::get<1>(p2).v == 42.0f);
  }

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(NV_IS_HOST, ({
                 const std::complex<float> p{1.0f, 42.0f};
                 cuda::std::pair<ExplicitConstructibleFrom<float>, ConvertibleFrom<float>> p2{p};
                 assert(cuda::std::get<0>(p2).v == 1.0f);
                 assert(cuda::std::get<1>(p2).v == 42.0f);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L

  // const overload should be called
  {
    const cuda::std::tuple<TracedCopyMove, TracedCopyMove> p{};
    cuda::std::pair<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> p2 = p;
    assert(constCopyCtrCalled(cuda::std::get<0>(p2).v));
    assert(constCopyCtrCalled(cuda::std::get<1>(p2)));
  }

  {
    const cuda::std::array<TracedCopyMove, 2> p{};
    cuda::std::pair<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> p2 = p;
    assert(constCopyCtrCalled(cuda::std::get<0>(p2).v));
    assert(constCopyCtrCalled(cuda::std::get<1>(p2)));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
