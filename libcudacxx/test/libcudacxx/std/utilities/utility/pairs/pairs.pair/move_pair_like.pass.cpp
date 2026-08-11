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
#include "MoveOnly.h"
#include "test_macros.h"

TEST_FUNC constexpr bool test()
{
  // test implicit conversions.
  {
    using Pair = cuda::std::pair<ConvertibleFrom<MoveOnly>, ConvertibleFrom<int>>;
    cuda::std::tuple<MoveOnly, int> p{1, 2};
    Pair p2 = cuda::std::move(p);
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 2);
    static_assert(cuda::std::is_nothrow_constructible_v<Pair, cuda::std::tuple<MoveOnly, int>&&>);
  }

  { // Ensure we properly detect noexcept
    using Pair = cuda::std::pair<ConvertibleFrom<MoveOnly, false>, ConvertibleFrom<int>>;
    cuda::std::tuple<MoveOnly, int> p{1, 2};
    Pair p2 = cuda::std::move(p);
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 2);
    static_assert(!cuda::std::is_nothrow_constructible_v<Pair, cuda::std::tuple<MoveOnly, int>&&>);
  }

  { // Ensure we properly detect noexcept
    using Pair = cuda::std::pair<ConvertibleFrom<MoveOnly>, ConvertibleFrom<int, false>>;
    cuda::std::tuple<MoveOnly, int> p{1, 2};
    Pair p2 = cuda::std::move(p);
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 2);
    static_assert(!cuda::std::is_nothrow_constructible_v<Pair, cuda::std::tuple<MoveOnly, int>&&>);
  }

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::tuple<MoveOnly, int> p{1, 2};
                 cuda::std::pair<ConvertibleFrom<MoveOnly>, ConvertibleFrom<int>> p2 = cuda::std::move(p);
                 assert(cuda::std::get<0>(p2).v == 1);
                 assert(cuda::std::get<1>(p2).v == 2);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)

  {
    cuda::std::array<MoveOnly, 2> p{1, 42};
    cuda::std::pair<ConvertibleFrom<MoveOnly>, ConvertibleFrom<MoveOnly>> p2 = cuda::std::move(p);
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 42);
  }

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::array<MoveOnly, 2> p{1, 42};
                 cuda::std::pair<ConvertibleFrom<MoveOnly>, ConvertibleFrom<MoveOnly>> p2 = cuda::std::move(p);
                 assert(cuda::std::get<0>(p2).v == 1);
                 assert(cuda::std::get<1>(p2).v == 42);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)

  {
    cuda::std::complex<float> p{1.0f, 42.0f};
    cuda::std::pair<ConvertibleFrom<float>, ConvertibleFrom<float>> p2 = cuda::std::move(p);
    assert(cuda::std::get<0>(p2).v == 1.0f);
    assert(cuda::std::get<1>(p2).v == 42.0f);
  }

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::complex<float> p{1.0f, 42.0f};
                 cuda::std::pair<ConvertibleFrom<float>, ConvertibleFrom<float>> p2 = cuda::std::move(p);
                 assert(cuda::std::get<0>(p2).v == 1.0f);
                 assert(cuda::std::get<1>(p2).v == 42.0f);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L

  // test explicit conversions.
  {
    using Pair = cuda::std::pair<ExplicitConstructibleFrom<MoveOnly>, ExplicitConstructibleFrom<int>>;
    cuda::std::pair<MoveOnly, int> p{1, 2};
    Pair p2{cuda::std::move(p)};
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 2);
  }

  { // Ensure we properly detect noexcept
    using Pair = cuda::std::pair<ExplicitConstructibleFrom<MoveOnly, false>, ConvertibleFrom<int>>;
    cuda::std::tuple<MoveOnly, int> p{1, 2};
    Pair p2{cuda::std::move(p)};
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 2);
    static_assert(!cuda::std::is_nothrow_constructible_v<Pair, cuda::std::tuple<MoveOnly, int>&&>);
  }

  { // Ensure we properly detect noexcept
    using Pair = cuda::std::pair<ExplicitConstructibleFrom<MoveOnly>, ConvertibleFrom<int, false>>;
    cuda::std::tuple<MoveOnly, int> p{1, 2};
    Pair p2{cuda::std::move(p)};
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 2);
    static_assert(!cuda::std::is_nothrow_constructible_v<Pair, cuda::std::tuple<MoveOnly, int>&&>);
  }

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::pair<MoveOnly, int> p{1, 2};
                 cuda::std::pair<ExplicitConstructibleFrom<MoveOnly>, ExplicitConstructibleFrom<int>> p2{
                   cuda::std::move(p)};
                 assert(cuda::std::get<0>(p2).v == 1);
                 assert(cuda::std::get<1>(p2).v == 2);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)

  {
    cuda::std::array<MoveOnly, 2> p{1, 42};
    cuda::std::pair<ExplicitConstructibleFrom<MoveOnly>, ConvertibleFrom<MoveOnly>> p2{cuda::std::move(p)};
    assert(cuda::std::get<0>(p2).v == 1);
    assert(cuda::std::get<1>(p2).v == 42);
  }

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::array<MoveOnly, 2> p{1, 42};
                 cuda::std::pair<ExplicitConstructibleFrom<MoveOnly>, ConvertibleFrom<MoveOnly>> p2{cuda::std::move(p)};
                 assert(cuda::std::get<0>(p2).v == 1);
                 assert(cuda::std::get<1>(p2).v == 42);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)

  {
    cuda::std::complex<float> p{1.0f, 42.0f};
    cuda::std::pair<ExplicitConstructibleFrom<float>, ConvertibleFrom<float>> p2{cuda::std::move(p)};
    assert(cuda::std::get<0>(p2).v == 1.0f);
    assert(cuda::std::get<1>(p2).v == 42.0f);
  }

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::complex<float> p{1.0f, 42.0f};
                 cuda::std::pair<ExplicitConstructibleFrom<float>, ConvertibleFrom<float>> p2{cuda::std::move(p)};
                 assert(cuda::std::get<0>(p2).v == 1.0f);
                 assert(cuda::std::get<1>(p2).v == 42.0f);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L

  // overload should be called
  {
    cuda::std::tuple<TracedCopyMove, TracedCopyMove> p{};
    cuda::std::pair<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> p2 = cuda::std::move(p);
    assert(moveCtrCalled(cuda::std::get<0>(p2).v));
    assert(moveCtrCalled(cuda::std::get<1>(p2)));
  }

  {
    cuda::std::array<TracedCopyMove, 2> p{};
    cuda::std::pair<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> p2 = cuda::std::move(p);
    assert(moveCtrCalled(cuda::std::get<0>(p2).v));
    assert(moveCtrCalled(cuda::std::get<1>(p2)));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
