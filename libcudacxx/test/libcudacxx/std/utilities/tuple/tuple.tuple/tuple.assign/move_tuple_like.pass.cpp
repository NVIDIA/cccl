//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class TupleLike>
//   tuple& operator=(TupleLike&& u);

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
  {
    cuda::std::array<MoveOnly, 0> t0{};
    cuda::std::tuple<> t1{};
    t1 = cuda::std::move(t0);
  }

  {
    cuda::std::array<MoveOnly, 1> t0{MoveOnly(3)};
    cuda::std::tuple<MoveOnly> t1{};
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 3);
  }

  {
    cuda::std::array<MoveOnly, 2> t0{MoveOnly(3), MoveOnly(42)};
    cuda::std::tuple<MoveOnly, MoveOnly> t1{};
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 3);
    assert(cuda::std::get<1>(t1) == 42);
  }

  {
    cuda::std::array<MoveOnly, 3> t0{MoveOnly(3), MoveOnly(42), MoveOnly(1337)};
    cuda::std::tuple<MoveOnly, MoveOnly, MoveOnly> t1{};
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 3);
    assert(cuda::std::get<1>(t1) == 42);
    assert(cuda::std::get<2>(t1) == 1337);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(
    NV_IS_HOST,
    (

      {
        std::array<MoveOnly, 1> t0{MoveOnly(3)};
        cuda::std::tuple<MoveOnly> t1{};
        t1 = cuda::std::move(t0);
        assert(cuda::std::get<0>(t1) == 3);
      }

      {
        std::array<MoveOnly, 2> t0{MoveOnly(3), MoveOnly(42)};
        cuda::std::tuple<MoveOnly, MoveOnly> t1{};
        t1 = cuda::std::move(t0);
        assert(cuda::std::get<0>(t1) == 3);
        assert(cuda::std::get<1>(t1) == 42);
      }

      {
        std::array<MoveOnly, 3> t0{MoveOnly(3), MoveOnly(42), MoveOnly(1337)};
        cuda::std::tuple<MoveOnly, MoveOnly, MoveOnly> t1{};
        t1 = cuda::std::move(t0);
        assert(cuda::std::get<0>(t1) == 3);
        assert(cuda::std::get<1>(t1) == 42);
        assert(cuda::std::get<2>(t1) == 1337);
      }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    cuda::std::complex<float> t0(3.0f, 42.0f);
    cuda::std::tuple<float, float> t1{};
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 3.0f);
    assert(cuda::std::get<1>(t1) == 42.0f);
  }

  {
    cuda::std::complex<float> t0(3.0f, 42.0f);
    cuda::std::tuple<double, double> t1{};
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 3.0f);
    assert(cuda::std::get<1>(t1) == 42.0f);
  }

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      {
        std::complex<float> t0(3.0f, 42.0f);
        cuda::std::tuple<float, float> t1{};
        t1 = cuda::std::move(t0);
        assert(cuda::std::get<0>(t1) == 3.0f);
        assert(cuda::std::get<1>(t1) == 42.0f);
      }

      {
        std::complex<float> t0(3.0f, 42.0f);
        cuda::std::tuple<double, double> t1{};
        t1 = cuda::std::move(t0);
        assert(cuda::std::get<0>(t1) == 3.0f);
        assert(cuda::std::get<1>(t1) == 42.0f);
      }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      {
        std::tuple<> t0{};
        cuda::std::tuple<> t1{};
        t1 = cuda::std::move(t0);
      }

      {
        std::tuple<cuda::std::int8_t, MoveOnly> t0{cuda::std::int8_t{3}, MoveOnly(42)};
        cuda::std::tuple<short, MoveOnly> t1{};
        t1 = cuda::std::move(t0);
        assert(cuda::std::get<0>(t1) == 3);
        assert(cuda::std::get<1>(t1) == 42);
      }

      {
        std::pair<cuda::std::int8_t, MoveOnly> t0{cuda::std::int8_t{3}, MoveOnly(42)};
        cuda::std::tuple<short, MoveOnly> t1{};
        t1 = cuda::std::move(t0);
        assert(cuda::std::get<0>(t1) == 3);
        assert(cuda::std::get<1>(t1) == 42);
      }))
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)

  // Ensure that the right constructor was called
  {
    cuda::std::tuple<MoveAssign, TracedAssignment> t1{1};
    cuda::std::tuple<AssignableFrom<MoveAssign>, AssignableFrom<TracedAssignment>> t2{3};
    t2 = cuda::std::move(t1);
    assert(cuda::std::get<0>(t2).v.val == 1);
    assert(cuda::std::get<1>(t2).v.moveAssign == 1);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
